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
import time

# ===== Configuration =====
class Config:
    # Core Settings
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    SYMBOLS_FILE = "under_100rs_stocks.csv"
    
    # Yahoo Finance Settings
    YFINANCE_RATE_LIMIT = 2  # Requests per second
    VALIDATION_TIMEOUT = 10  # Seconds between batches
    
    # Trading Parameters
    RISK_PER_TRADE = 0.02
    MIN_RISK_REWARD = 2.0
    POSITION_SIZE_DAYS = 30
    
    # Technical Parameters
    RSI_OVERSOLD = 30
    RSI_OVERBOUGHT = 70
    VOLUME_SPIKE = 1.8

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
        self.portfolio = {}
        self.market_tz = ZoneInfo("Asia/Kolkata")
        self.paused = False
        self.blacklisted_symbols = set()
        self.valid_symbols = []
        self.last_request_time = 0
        self.request_counter = 0

    # ===== Rate Limited Requests =====
    async def _rate_limited_request(self):
        """Enforce rate limiting for Yahoo Finance API"""
        current_time = time.time()
        elapsed = current_time - self.last_request_time
        
        # Reset counter if more than 1 second has passed
        if elapsed >= 1:
            self.request_counter = 0
            self.last_request_time = current_time
        
        # Wait if we've hit the rate limit
        if self.request_counter >= Config.YFINANCE_RATE_LIMIT:
            wait_time = 1 - elapsed
            if wait_time > 0:
                await asyncio.sleep(wait_time)
            self.request_counter = 0
            self.last_request_time = time.time()
        
        self.request_counter += 1

    # ===== Symbol Handling =====
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

    async def _validate_symbol(self, symbol: str) -> bool:
        """Validate a single symbol with multiple approaches"""
        try:
            await self._rate_limited_request()
            
            # Approach 1: Quick price check
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")
            
            if not hist.empty and not hist['Close'].isna().all():
                return True
                
            # Approach 2: Full data check if first fails
            await self._rate_limited_request()
            data = yf.download(symbol, period="1d", progress=False)
            
            if not data.empty and not data['Close'].isna().all():
                return True
                
            logger.warning(f"No valid data for {symbol}")
            return False
            
        except Exception as e:
            logger.warning(f"Validation failed for {symbol}: {str(e)}")
            return False

    async def _validate_symbols_batch(self):
        """Validate symbols in batches with rate limiting"""
        self.valid_symbols = []
        batch_size = 10
        delay = Config.VALIDATION_TIMEOUT / (len(self.symbols) / batch_size)
        
        for i in range(0, len(self.symbols), batch_size):
            batch = self.symbols[i:i + batch_size]
            tasks = [self._validate_symbol(symbol) for symbol in batch]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for symbol, is_valid in zip(batch, results):
                if is_valid and not isinstance(is_valid, Exception):
                    self.valid_symbols.append(symbol)
                    logger.info(f"✅ Validated {symbol}")
                else:
                    self.blacklisted_symbols.add(symbol)
                    logger.warning(f"❌ Failed to validate {symbol}")
            
            if i + batch_size < len(self.symbols):
                await asyncio.sleep(delay)

    # ===== Data Fetching =====
    async def _fetch_data(self, symbol: str, interval: str) -> Optional[pd.DataFrame]:
        """Fetch market data with rate limiting"""
        if symbol in self.blacklisted_symbols:
            return None

        try:
            await self._rate_limited_request()
            
            # Try direct download first
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
            await self._rate_limited_request()
            ticker = await asyncio.to_thread(yf.Ticker, symbol)
            hist = await asyncio.to_thread(
                ticker.history,
                period="7d" if interval != "1d" else "30d",
                interval=interval
            )
            
            return hist if not hist.empty else None
            
        except Exception as e:
            logger.error(f"Failed to fetch {symbol}: {str(e)}")
            self.blacklisted_symbols.add(symbol)
            return None

    # ===== Trading Logic =====
    async def _analyze_symbol(self, symbol: str) -> Optional[Dict]:
        """Analyze symbol for trading opportunities"""
        daily, intraday = await asyncio.gather(
            self._fetch_data(symbol, "1d"),
            self._fetch_data(symbol, "15m")
        )
        
        if daily is None or intraday is None:
            return None

        try:
            # Technical Indicators
            atr = ta.atr(intraday['High'], intraday['Low'], intraday['Close'], length=14).iloc[-1]
            rsi = ta.rsi(intraday['Close'], length=14).iloc[-1]
            macd = ta.macd(intraday['Close']).iloc[-1]
            
            # Price Action
            price = intraday['Close'].iloc[-1]
            stop_loss, take_profit = entry - (1.5 * atr), entry + (3 * atr)
            risk_reward = (take_profit - price) / (price - stop_loss)

            # Signal Conditions
            buy_conditions = [
                rsi < Config.RSI_OVERSOLD,
                macd['MACD_12_26_9'] > macd['MACDs_12_26_9'],
                price > daily['Close'].rolling(50).mean().iloc[-1] > daily['Close'].rolling(200).mean().iloc[-1],
                intraday['Volume'].iloc[-1] > (Config.VOLUME_SPIKE * intraday['Volume'].rolling(20).mean().iloc[-1]),
                risk_reward >= Config.MIN_RISK_REWARD
            ]

            if sum(buy_conditions) >= 4:
                volatility = daily['Close'].pct_change().std()
                position_size = (10000 * Config.RISK_PER_TRADE) / ((price - stop_loss) * volatility)
                
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
        except Exception as e:
            logger.error(f"Analysis error for {symbol}: {str(e)}")
        
        return None

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
            f"💰 Entry: ₹{signal['price']:.2f}",
            f"🛑 Stop-Loss: ₹{signal['stop_loss']:.2f}",
            f"🎯 Take-Profit: ₹{signal['take_profit']:.2f}",
            f"📊 Size: {signal['size']} shares",
            "",
            "<b>Rationale:</b>",
            *signal['rationale']
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
        """Telegram status command"""
        status = [
            f"📊 <b>Bot Status</b>",
            f"• Valid Symbols: {len(self.valid_symbols)}",
            f"• Active Positions: {len(self.portfolio)}",
            f"• Market: {'Open' if self._is_market_open() else 'Closed'}"
        ]
        await update.message.reply_text("\n".join(status), parse_mode='HTML')

    # ===== Main Loop =====
    async def run(self):
        """Complete trading workflow"""
        try:
            self.app = Application.builder().token(Config.TELEGRAM_TOKEN).build()
            self.app.add_handler(CommandHandler("status", self._get_status))
            
            await self._send_telegram("🤖 Trading Bot Started")
            
            # Validate symbols with rate limiting
            logger.info("Starting symbol validation...")
            await self._validate_symbols_batch()
            
            if not self.valid_symbols:
                await self._send_telegram("❌ No valid symbols found! Check logs.")
                return
            
            await self._send_telegram(f"✅ Ready with {len(self.valid_symbols)} valid symbols")
            
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
