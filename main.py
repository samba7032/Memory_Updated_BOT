import os
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import asyncio
import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from datetime import datetime, time as dtime
from zoneinfo import ZoneInfo
import logging
from typing import List, Dict, Optional, Tuple
import sys

# ===== Configuration =====
class Config:
    # Core Settings
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    SYMBOLS_FILE = "under_100rs_stocks.csv"
    
    # Trading Parameters
    RISK_PER_TRADE = 0.02  # 2% of capital per trade
    MIN_RISK_REWARD = 2.0  # 1:2 minimum
    POSITION_SIZE_DAYS = 30  # For volatility calculation
    
    # Technical Parameters
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    VOLUME_SPIKE = 1.8  # 180% of average volume

# ===== Logging =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('bot.log')]
)
logger = logging.getLogger(__name__)

class ProfessionalTradingBot:
    def __init__(self):
        self.app = None
        self.client = httpx.AsyncClient(timeout=30.0)
        self.symbols = self._load_symbols()
        self.portfolio = {}  # Track live positions
        self.market_tz = ZoneInfo("Asia/Kolkata")
        self.paused = False
        self.blacklisted_symbols = set()
        self.valid_symbols = []
        self.symbol_validation_attempts = 2

    # ===== Core Methods =====
    def _is_market_open(self) -> bool:
        """Check if market is open (9:15 AM - 3:30 PM IST, Mon-Fri)"""
        now = datetime.now(self.market_tz)
        return (now.weekday() < 5 and 
                dtime(9, 15) <= now.time() <= dtime(15, 30))

    def _load_symbols(self) -> List[str]:
        """Load and clean symbols from CSV"""
        try:
            df = pd.read_csv(Config.SYMBOLS_FILE)
            symbols = [f"{s.strip().upper()}.NS" for s in df['Symbol'].dropna()]
            logger.info(f"Loaded {len(symbols)} symbols from CSV")
            return symbols
        except Exception as e:
            logger.critical(f"Failed to load symbols: {e}")
            sys.exit(1)

    async def _fetch_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch market data with robust error handling"""
        if symbol in self.blacklisted_symbols:
            return None

        try:
            # First try standard download
            data = await asyncio.to_thread(
                yf.download,
                symbol,
                period="7d" if interval != "1d" else "30d",
                interval=interval,
                progress=False,
                threads=False
            )
            
            if not data.empty:
                return data
            
            # Fallback to Ticker method
            ticker = await asyncio.to_thread(yf.Ticker, symbol)
            hist = await asyncio.to_thread(
                ticker.history,
                period="7d" if interval != "1d" else "30d",
                interval=interval
            )
            
            if not hist.empty:
                return hist
                
            logger.warning(f"No data found for {symbol}")
            self.blacklisted_symbols.add(symbol)
            return None

        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {str(e)}")
            self.blacklisted_symbols.add(symbol)
            return None

    async def _validate_symbols(self):
        """Validate symbols with quick checks"""
        self.valid_symbols = []
        
        for symbol in self.symbols:
            try:
                # Quick check using Ticker.info
                ticker = await asyncio.to_thread(yf.Ticker, symbol)
                info = await asyncio.to_thread(getattr, ticker, 'info', {})
                
                if info.get('regularMarketPrice'):
                    self.valid_symbols.append(symbol)
                    logger.info(f"✅ Validated {symbol}")
                    continue
                    
                # Full check if quick check failed
                data = await self._fetch_data(symbol, "1d")
                if data is not None and not data.empty:
                    self.valid_symbols.append(symbol)
                else:
                    logger.warning(f"❌ Invalid symbol: {symbol}")
                    self.blacklisted_symbols.add(symbol)
                    
            except Exception as e:
                logger.error(f"Validation error for {symbol}: {str(e)}")
                self.blacklisted_symbols.add(symbol)
            
            await asyncio.sleep(0.5)  # Rate limiting

        logger.info(f"Validation complete. Valid: {len(self.valid_symbols)}, Blacklisted: {len(self.blacklisted_symbols)}")
        
        if not self.valid_symbols:
            error_msg = "❌ No valid symbols found! Check your symbols file."
            logger.error(error_msg)
            await self._send_telegram(error_msg)

    # ===== Trading Logic =====
    async def _analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Analyze symbol for trading opportunities"""
        daily, intraday = await asyncio.gather(
            self._fetch_data(symbol, "1d"),
            self._fetch_data(symbol, "15m")
        )
        
        if daily is None or intraday is None:
            return None

        # Technical Indicators
        atr = ta.atr(intraday['High'], intraday['Low'], intraday['Close'], length=14).iloc[-1]
        rsi = ta.rsi(intraday['Close'], length=14).iloc[-1]
        macd = ta.macd(intraday['Close']).iloc[-1]
        
        # Price Action
        price = intraday['Close'].iloc[-1]
        stop_loss, take_profit = self._calculate_risk(price, atr)
        risk_reward = (take_profit - price) / (price - stop_loss)

        # Signal Conditions
        buy_conditions = [
            rsi < Config.RSI_OVERSOLD,
            macd['MACD_12_26_9'] > macd['MACDs_12_26_9'],
            self._check_trend_alignment(daily),
            self._check_volume_spike(intraday),
            risk_reward >= Config.MIN_RISK_REWARD
        ]

        if sum(buy_conditions) >= 4:  # Require 4/5 conditions
            position_size = self._calculate_position_size(symbol, price, stop_loss)
            return {
                'symbol': symbol,
                'action': 'BUY',
                'price': price,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'size': int(position_size),
                'rationale': [
                    f"RSI: {rsi:.1f}",
                    f"ATR: {atr:.2f}",
                    f"Risk/Reward: 1:{risk_reward:.1f}",
                    f"Size: {int(position_size)} shares"
                ]
            }
        return None

    # ===== Utility Methods =====
    def _check_trend_alignment(self, data: pd.DataFrame) -> bool:
        sma50 = ta.sma(data['Close'], length=50).iloc[-1]
        sma200 = ta.sma(data['Close'], length=200).iloc[-1]
        return data['Close'].iloc[-1] > sma50 > sma200

    def _check_volume_spike(self, data: pd.DataFrame) -> bool:
        return data['Volume'].iloc[-1] > (Config.VOLUME_SPIKE * data['Volume'].rolling(20).mean().iloc[-1])

    def _calculate_risk(self, entry: float, atr: float) -> Tuple[float, float]:
        stop_loss = entry - (1.5 * atr)
        take_profit = entry + (3 * atr)
        return stop_loss, take_profit

    def _calculate_position_size(self, symbol: str, entry: float, stop_loss: float) -> float:
        hist_data = yf.download(symbol, period=f"{Config.POSITION_SIZE_DAYS}d")['Close']
        volatility = hist_data.pct_change().std()
        risk_amount = 10000 * Config.RISK_PER_TRADE
        return risk_amount / (entry - stop_loss) / volatility

    # ===== Execution Methods =====
    async def _execute_trade(self, signal: Dict):
        """Execute trade with risk management"""
        self.portfolio[signal['symbol']] = {
            'entry': signal['price'],
            'sl': signal['stop_loss'],
            'tp': signal['take_profit'],
            'size': signal['size'],
            'entry_time': datetime.now()
        }
        
        message = [
            f"🚀 <b>{signal['action']} {signal['symbol']}</b>",
            f"🔢 Size: {signal['size']} shares",
            f"💰 Entry: ₹{signal['price']:.2f}",
            f"🛑 Stop-Loss: ₹{signal['stop_loss']:.2f}",
            f"🎯 Take-Profit: ₹{signal['take_profit']:.2f}",
            "",
            "<b>Rationale:</b>",
            *signal['rationale']
        ]
        await self._send_telegram("\n".join(message))

    async def _manage_positions(self):
        """Check open positions for exits"""
        for symbol, trade in list(self.portfolio.items()):
            data = await self._fetch_data(symbol, "15m")
            if data is None:
                continue

            current_price = data['Close'].iloc[-1]
            
            if current_price <= trade['sl'] or current_price >= trade['tp']:
                await self._close_position(symbol, current_price)
            elif current_price > trade['entry'] * 1.03:
                new_sl = max(current_price * 0.98, trade['entry'] * 1.03)
                if new_sl > trade['sl']:
                    self.portfolio[symbol]['sl'] = new_sl

    async def _close_position(self, symbol: str, exit_price: float):
        """Handle position closing"""
        trade = self.portfolio.pop(symbol)
        pnl = (exit_price - trade['entry']) * trade['size']
        
        message = [
            f"🔴 <b>CLOSE {symbol}</b>",
            f"💰 Exit: ₹{exit_price:.2f}",
            f"📈 P&L: ₹{pnl:.2f} ({pnl/(trade['entry']*trade['size']):.1%})",
            f"⏱️ Duration: {(datetime.now() - trade['entry_time']).seconds//60} mins"
        ]
        await self._send_telegram("\n".join(message))

    # ===== Telegram Methods =====
    async def _send_telegram(self, text: str):
        """Send message with error handling"""
        try:
            await self.app.bot.send_message(
                chat_id=Config.TELEGRAM_CHAT_ID,
                text=text,
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")

    async def _get_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Telegram command handler"""
        status = [
            f"📊 <b>Bot Status</b>",
            f"• Valid Symbols: {len(self.valid_symbols)}",
            f"• Positions: {len(self.portfolio)}",
            f"• Market: {'Open' if self._is_market_open() else 'Closed'}"
        ]
        await update.message.reply_text("\n".join(status), parse_mode='HTML')

    async def _list_symbols(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List all symbols"""
        msg = "📜 <b>Symbol Status</b>\n"
        msg += f"✅ Valid: {len(self.valid_symbols)}\n"
        msg += f"❌ Blacklisted: {len(self.blacklisted_symbols)}"
        await update.message.reply_text(msg, parse_mode='HTML')

    # ===== Main Loop =====
    async def run(self):
        """Complete trading workflow"""
        try:
            self.app = Application.builder().token(Config.TELEGRAM_TOKEN).build()
            self.app.add_handler(CommandHandler("status", self._get_status))
            self.app.add_handler(CommandHandler("symbols", self._list_symbols))
            
            await self._send_telegram("🤖 Trading Bot Started")
            
            # Initial validation
            await self._validate_symbols()
            
            if not self.valid_symbols:
                await self._send_telegram("🛑 No valid symbols available. Exiting.")
                return
            
            # Main trading loop
            while True:
                try:
                    if not self._is_market_open():
                        await asyncio.sleep(60)
                        continue
                        
                    for symbol in self.valid_symbols:
                        try:
                            signal = await self._analyze_symbol(symbol)
                            if signal:
                                await self._execute_trade(signal)
                                await asyncio.sleep(1)
                        except Exception as e:
                            logger.error(f"Error processing {symbol}: {str(e)}")
                            
                    await self._manage_positions()
                    await asyncio.sleep(300)
                    
                except Exception as e:
                    logger.error(f"Main loop error: {str(e)}")
                    await asyncio.sleep(60)
                    
        except Exception as e:
            logger.critical(f"Fatal error: {str(e)}")
            await self._send_telegram(f"💀 Bot crashed: {str(e)}")

async def main():
    bot = ProfessionalTradingBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
