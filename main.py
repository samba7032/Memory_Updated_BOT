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
        self.blacklisted_symbols = set()  # Track symbols that consistently fail
        self.symbol_validation_attempts = 3  # Number of attempts to validate symbols

    # ===== 8 Profit Techniques =====
    
    # 1. Multi-Timeframe Analysis
    async def _get_multi_tf_data(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Fetch daily + intraday data"""
        daily = await self._fetch_data(symbol, "1d")
        intraday = await self._fetch_data(symbol, "15m")
        return daily, intraday

    # 2. Volume-Weighted Signals
    def _check_volume_spike(self, data: pd.DataFrame) -> bool:
        return data['Volume'].iloc[-1] > (Config.VOLUME_SPIKE * data['Volume'].rolling(20).mean().iloc[-1])

    # 3. Smart Position Sizing
    def _calculate_position_size(self, symbol: str, entry: float, stop_loss: float) -> float:
        """Calculate shares based on volatility"""
        hist_data = yf.download(symbol, period=f"{Config.POSITION_SIZE_DAYS}d")['Close']
        volatility = hist_data.pct_change().std()
        risk_amount = 10000 * Config.RISK_PER_TRADE  # Example ₹10k capital
        return risk_amount / (entry - stop_loss) / volatility

    # 4. Trend Confirmation
    def _check_trend_alignment(self, daily_data: pd.DataFrame) -> bool:
        sma50 = ta.sma(daily_data['Close'], length=50).iloc[-1]
        sma200 = ta.sma(daily_data['Close'], length=200).iloc[-1]
        return daily_data['Close'].iloc[-1] > sma50 > sma200  # Bullish stack

    # 5. Candlestick Patterns
    def _detect_candlestick_patterns(self, data: pd.DataFrame) -> List[str]:
        patterns = []
        latest = data.iloc[-3:]  # Last 3 candles
        
        # Bullish Engulfing
        if (latest.iloc[-2]['Close'] < latest.iloc[-2]['Open'] and 
            latest.iloc[-1]['Close'] > latest.iloc[-1]['Open'] and
            latest.iloc[-1]['Close'] > latest.iloc[-2]['Open']):
            patterns.append("Bullish Engulfing")
            
        return patterns

    # 6. Risk-Managed Entries
    def _calculate_risk(self, entry: float, atr: float) -> Tuple[float, float]:
        stop_loss = entry - (1.5 * atr)
        take_profit = entry + (3 * atr)
        return stop_loss, take_profit

    # 7. News/Sentiment Filter (Placeholder)
    async def _check_news_sentiment(self, symbol: str) -> bool:
        """Integrate with NewsAPI or similar"""
        return True  # Placeholder

    # 8. Dynamic Exit Strategy
    def _trailing_stop(self, current_price: float, entry: float) -> float:
        return max(current_price * 0.98, entry * 1.03)  # 2% trail or 3% profit lock

    # ===== Core Methods =====
    def _is_market_open(self) -> bool:
        """Check if market is open (9:15 AM - 3:30 PM IST, Mon-Fri)"""
        now = datetime.now(self.market_tz)
        return (now.weekday() < 5 and 
                dtime(9, 15) <= now.time() <= dtime(15, 30))

    async def _manage_positions(self):
        """Check open positions for exits"""
        for symbol, trade in list(self.portfolio.items()):
            data = await self._fetch_data(symbol, "15m")
            if data is None:
                continue

            current_price = data['Close'].iloc[-1]
            
            # Check stop-loss or take-profit
            if current_price <= trade['sl'] or current_price >= trade['tp']:
                await self._close_position(symbol, current_price)
            
            # Update trailing stop
            elif current_price > trade['entry'] * 1.03:  # After 3% profit
                new_sl = self._trailing_stop(current_price, trade['entry'])
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

    # ===== Trading Logic =====
    async def _analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Complete analysis with all 8 techniques"""
        if symbol in self.blacklisted_symbols:
            return None

        daily, intraday = await self._get_multi_tf_data(symbol)
        if daily is None or intraday is None:
            return None

        # Technical Indicators
        atr = ta.atr(intraday['High'], intraday['Low'], intraday['Close'], length=14).iloc[-1]
        rsi = ta.rsi(intraday['Close'], length=14).iloc[-1]
        macd = ta.macd(intraday['Close']).iloc[-1]
        patterns = self._detect_candlestick_patterns(intraday)

        # Price Action
        price = intraday['Close'].iloc[-1]
        stop_loss, take_profit = self._calculate_risk(price, atr)
        risk_reward = (take_profit - price) / (price - stop_loss)

        # Signal Generation
        buy_conditions = [
            rsi < Config.RSI_OVERSOLD,
            macd['MACD_12_26_9'] > macd['MACDs_12_26_9'],
            self._check_trend_alignment(daily),
            self._check_volume_spike(intraday),
            "Bullish Engulfing" in patterns,
            await self._check_news_sentiment(symbol),
            risk_reward >= Config.MIN_RISK_REWARD
        ]

        if sum(buy_conditions) >= 5:  # Require 5/7 conditions
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
                ] + patterns
            }
        return None

    # ===== Execution =====
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

    # ===== Utilities =====
    def _load_symbols(self) -> List[str]:
        """Load symbols from CSV with enhanced validation"""
        try:
            # Read CSV and clean symbols
            df = pd.read_csv(Config.SYMBOLS_FILE)
            raw_symbols = [s.strip().upper() for s in df['Symbol'].dropna()]
            
            # Remove duplicates and empty strings
            unique_symbols = list(set(s for s in raw_symbols if s))
            
            # Add .NS suffix if not present, but don't double it
            processed_symbols = []
            for symbol in unique_symbols:
                if not symbol.endswith('.NS'):
                    processed_symbols.append(f"{symbol}.NS")
                else:
                    processed_symbols.append(symbol)
            
            logger.info(f"Loaded {len(processed_symbols)} symbols from CSV")
            return processed_symbols
            
        except Exception as e:
            logger.critical(f"Failed to load symbols: {e}")
            sys.exit(1)

    async def _fetch_data(self, symbol: str, interval: str, retries: int = 3) -> Optional[pd.DataFrame]:
        """Fetch market data with enhanced error handling"""
        if symbol in self.blacklisted_symbols:
            return None

        for attempt in range(retries):
            try:
                logger.info(f"Fetching {symbol} ({interval}), attempt {attempt + 1}/{retries}")
                
                # Use smaller period for intraday data to reduce load
                period = "60d" if interval == "1d" else "7d"
                
                data = await asyncio.to_thread(
                    yf.download,
                    symbol,
                    period=period,
                    interval=interval,
                    progress=False,
                    threads=False  # Disable parallel downloads to reduce errors
                )
                
                if data.empty:
                    logger.warning(f"Empty data for {symbol}, attempt {attempt + 1}/{retries}")
                    if attempt == retries - 1:
                        self.blacklisted_symbols.add(symbol)
                        logger.error(f"Adding {symbol} to blacklist - empty data")
                    continue
                
                # Check for all NA values
                if data['Close'].isna().all() or data['Volume'].isna().all():
                    logger.warning(f"Invalid data (all NA) for {symbol}, attempt {attempt + 1}/{retries}")
                    if attempt == retries - 1:
                        self.blacklisted_symbols.add(symbol)
                        logger.error(f"Adding {symbol} to blacklist - invalid data")
                    continue
                
                return data

            except Exception as e:
                logger.warning(f"Data fetch failed for {symbol}: {str(e)}, attempt {attempt + 1}/{retries}")
                if attempt == retries - 1:
                    self.blacklisted_symbols.add(symbol)
                    logger.error(f"Adding {symbol} to blacklist after failed retries")
            
            await asyncio.sleep(2)  # Wait before retry
        
        return None

    async def _validate_symbols(self):
        """Validate all symbols with detailed logging"""
        valid_symbols = []
        
        logger.info(f"Starting validation of {len(self.symbols)} symbols...")
        
        for symbol in self.symbols:
            validation_success = False
            
            for attempt in range(self.symbol_validation_attempts):
                try:
                    data = await self._fetch_data(symbol, "1d")
                    
                    if data is not None and not data.empty:
                        # Additional checks for valid data
                        last_close = data['Close'].iloc[-1]
                        last_volume = data['Volume'].iloc[-1]
                        
                        if pd.notna(last_close) and pd.notna(last_volume) and last_close > 0:
                            valid_symbols.append(symbol)
                            validation_success = True
                            logger.info(f"✅ Validated {symbol} (Close: {last_close:.2f}, Volume: {last_volume:,.0f})")
                            break
                        else:
                            logger.warning(f"Invalid data for {symbol} (Close: {last_close}, Volume: {last_volume})")
                    else:
                        logger.warning(f"No data returned for {symbol}")
                    
                except Exception as e:
                    logger.error(f"Validation error for {symbol}: {str(e)}")
                
                await asyncio.sleep(1)  # Short delay between attempts
            
            if not validation_success:
                logger.warning(f"❌ Failed to validate {symbol} after {self.symbol_validation_attempts} attempts")
                self.blacklisted_symbols.add(symbol)
        
        self.symbols = valid_symbols
        logger.info(f"Validation complete. {len(valid_symbols)} valid symbols, {len(self.blacklisted_symbols)} blacklisted")
        
        if not valid_symbols:
            error_msg = "❌ CRITICAL: No valid symbols found! Check your symbols file and network connection."
            logger.error(error_msg)
            await self._send_telegram(error_msg)

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
            f"• Symbols: {len(self.symbols)}",
            f"• Active Positions: {len(self.portfolio)}",
            f"• Blacklisted Symbols: {len(self.blacklisted_symbols)}",
            f"• Market: {'Open' if self._is_market_open() else 'Closed'}"
        ]
        await update.message.reply_text("\n".join(status), parse_mode='HTML')

    async def _list_symbols(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List all valid and blacklisted symbols"""
        valid_msg = "✅ Valid Symbols:\n" + "\n".join(self.symbols) if self.symbols else "No valid symbols"
        blacklisted_msg = "❌ Blacklisted Symbols:\n" + "\n".join(self.blacklisted_symbols) if self.blacklisted_symbols else "No blacklisted symbols"
        
        await update.message.reply_text(
            f"{valid_msg}\n\n{blacklisted_msg}",
            parse_mode='HTML'
        )

    # ===== Main Loop =====
    async def run(self):
        """Complete trading workflow"""
        self.app = Application.builder().token(Config.TELEGRAM_TOKEN).build()
        self.app.add_handler(CommandHandler("status", self._get_status))
        self.app.add_handler(CommandHandler("symbols", self._list_symbols))
        
        await self._send_telegram("💼 Professional Bot Activated")
        
        # Validate symbols with detailed logging
        await self._validate_symbols()
        
        if not self.symbols:
            logger.error("No valid symbols available. Exiting.")
            return
        
        while True:
            try:
                if self.paused or not self._is_market_open():
                    await asyncio.sleep(60)
                    continue

                for symbol in self.symbols:
                    if symbol in self.blacklisted_symbols:
                        continue
                        
                    try:
                        signal = await self._analyze_symbol(symbol)
                        if signal:
                            await self._execute_trade(signal)
                            await asyncio.sleep(1)  # Rate limit
                    except Exception as e:
                        logger.error(f"Error processing {symbol}: {str(e)}")
                        if "No data found" in str(e):
                            self.blacklisted_symbols.add(symbol)

                await self._manage_positions()
                await asyncio.sleep(300)  # 5 min interval

            except Exception as e:
                logger.error(f"Main loop error: {str(e)}")
                await asyncio.sleep(60)

async def main():
    bot = ProfessionalTradingBot()
    await bot.run()

if __name__ == "__main__":
    asyncio.run(main())
