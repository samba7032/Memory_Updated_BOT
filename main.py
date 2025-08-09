import os
# Pre-configure environment for Railway
import sys
if 'railway' in os.getenv('RAILWAY_ENVIRONMENT_NAME', '').lower():
    os.environ['NO_NUMBA'] = '1'  # Disable numba for pandas-ta
    os.environ['PYTHONUNBUFFERED'] = '1'

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
import sys
from typing import Optional, Tuple, List

# ===== Configuration =====
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
SYMBOLS_FILE = "symbols.csv"  # Local file in repository
DATA_DAYS = 5  # Reduced lookback period for memory efficiency
CHECK_INTERVAL = 300  # 5 minutes between scans
MAX_CONCURRENT_REQUESTS = 3  # Conservative for Railway's free tier
RISK_REWARD_RATIO = 2

# ===== Logging Setup =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
logging.getLogger('httpx').setLevel(logging.WARNING)  # Reduce noise

class TradingBot:
    def __init__(self):
        self.app = None
        self.client = httpx.AsyncClient(timeout=30.0)
        self.symbols = self._load_symbols()
        self.failed_symbols = set()
        self.market_tz = ZoneInfo("Asia/Kolkata")
        self.paused = False
        self.signal_cache = {}

    def _load_symbols(self) -> List[str]:
        """Load and validate trading symbols"""
        try:
            if not os.path.exists(SYMBOLS_FILE):
                logger.critical(f"Symbols file not found: {SYMBOLS_FILE}")
                sys.exit(1)

            df = pd.read_csv(SYMBOLS_FILE)
            if 'Symbol' not in df.columns:
                logger.critical("CSV missing 'Symbol' column")
                sys.exit(1)

            symbols = [f"{s.strip().upper()}.NS" for s in df['Symbol'].dropna()]
            logger.info(f"Loaded {len(symbols)} symbols")
            return symbols
        except Exception as e:
            logger.critical(f"Symbol loading failed: {str(e)}")
            sys.exit(1)

    async def start(self):
        """Main entry point with initialization"""
        if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
            logger.critical("Missing Telegram credentials in environment variables")
            sys.exit(1)

        self.app = Application.builder().token(TELEGRAM_TOKEN).build()
        self.app.add_handler(CommandHandler("pause", self._pause_bot))
        self.app.add_handler(CommandHandler("resume", self._resume_bot))
        self.app.add_handler(CommandHandler("status", self._get_status))

        await self._send_telegram_message("🚀 Trading Bot Activated")
        await self._run_trading_loop()

    async def _run_trading_loop(self):
        """Continuous trading signal generation"""
        while True:
            try:
                if self.paused:
                    await asyncio.sleep(60)
                    continue

                if not self._is_market_open():
                    logger.debug("Market closed - sleeping")
                    await asyncio.sleep(CHECK_INTERVAL)
                    continue

                active_symbols = [s for s in self.symbols if s not in self.failed_symbols]
                logger.info(f"Scanning {len(active_symbols)} symbols")

                # Process in batches to avoid rate limits
                for i in range(0, len(active_symbols), MAX_CONCURRENT_REQUESTS):
                    batch = active_symbols[i:i + MAX_CONCURRENT_REQUESTS]
                    await self._process_batch(batch)
                    await asyncio.sleep(5)  # Rate limiting

                await asyncio.sleep(CHECK_INTERVAL)

            except Exception as e:
                logger.error(f"Trading loop error: {str(e)}")
                await asyncio.sleep(60)

    async def _process_batch(self, symbols: List[str]):
        """Process a batch of symbols concurrently"""
        tasks = [self._analyze_symbol(symbol) for symbol in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for symbol, result in zip(symbols, results):
            if isinstance(result, Exception):
                logger.warning(f"Analysis failed for {symbol}: {str(result)}")
                self.failed_symbols.add(symbol)
            elif result:
                signal, price, notes = result
                await self._send_trading_alert(signal, symbol, price, notes)

    async def _analyze_symbol(self, symbol: str) -> Optional[Tuple[str, float, List[str]]]:
        """Generate trading signal for a single symbol"""
        try:
            data = await self._fetch_stock_data(symbol)
            if data is None or data.empty:
                return None

            current_price = data['Close'].iloc[-1]
            rsi = ta.rsi(data['Close'], length=14).iloc[-1]
            macd = ta.macd(data['Close']).iloc[-1]
            sma20 = ta.sma(data['Close'], length=20).iloc[-1]

            notes = []
            buy_conditions = 0
            sell_conditions = 0

            # Buy signal conditions
            if rsi < 30:
                notes.append(f"RSI {rsi:.1f} (Oversold)")
                buy_conditions += 1
            if macd['MACD_12_26_9'] > macd['MACDs_12_26_9']:
                notes.append("MACD Bullish")
                buy_conditions += 1
            if current_price > sma20:
                notes.append("Price > SMA20")
                buy_conditions += 1

            # Sell signal conditions
            if rsi > 70:
                notes.append(f"RSI {rsi:.1f} (Overbought)")
                sell_conditions += 1
            if macd['MACD_12_26_9'] < macd['MACDs_12_26_9']:
                notes.append("MACD Bearish")
                sell_conditions += 1
            if current_price < sma20:
                notes.append("Price < SMA20")
                sell_conditions += 1

            if buy_conditions >= 2 and buy_conditions > sell_conditions:
                return ('BUY', current_price, notes)
            elif sell_conditions >= 2 and sell_conditions > buy_conditions:
                return ('SELL', current_price, notes)

            return None

        except Exception as e:
            logger.warning(f"Analysis error for {symbol}: {str(e)}")
            raise e

    async def _fetch_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Fetch stock data with retry logic"""
        try:
            interval = "15m" if self._is_market_open() else "1d"
            data = await asyncio.to_thread(
                yf.download,
                symbol,
                period=f"{DATA_DAYS}d",
                interval=interval,
                progress=False,
                auto_adjust=True
            )
            return data if not data.empty else None
        except Exception as e:
            logger.debug(f"Data fetch failed for {symbol}: {str(e)}")
            return None

    def _is_market_open(self) -> bool:
        """Check if Indian stock market is open"""
        now = datetime.now(self.market_tz)
        return (now.weekday() < 5 and  # Monday-Friday
                dtime(9, 15) <= now.time() <= dtime(15, 30))

    # ===== Telegram Methods =====
    async def _send_trading_alert(self, signal: str, symbol: str, price: float, notes: List[str]):
        """Send formatted trading alert"""
        emoji = "🟢" if signal == "BUY" else "🔴"
        message = [
            f"{emoji} <b>{signal} {symbol}</b> {emoji}",
            f"Price: ₹{price:.2f}",
            "",
            "<b>Rationale:</b>",
            *notes,
            "",
            f"<i>{datetime.now(self.market_tz).strftime('%Y-%m-%d %H:%M:%S')}</i>"
        ]
        await self._send_telegram_message("\n".join(message))

    async def _send_telegram_message(self, text: str):
        """Safe message sending with retry"""
        try:
            await self.app.bot.send_message(
                chat_id=TELEGRAM_CHAT_ID,
                text=text,
                parse_mode='HTML'
            )
        except Exception as e:
            logger.error(f"Telegram send failed: {str(e)}")

    # ===== Command Handlers =====
    async def _pause_bot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.paused = True
        await update.message.reply_text("⏸️ Bot paused")

    async def _resume_bot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        self.paused = False
        await update.message.reply_text("▶️ Bot resumed")

    async def _get_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        status = [
            f"🔍 <b>Bot Status</b>",
            f"• Symbols: {len(self.symbols)}",
            f"• Failed: {len(self.failed_symbols)}",
            f"• Market: {'Open' if self._is_market_open() else 'Closed'}",
            f"• State: {'Paused' if self.paused else 'Running'}"
        ]
        await update.message.reply_text("\n".join(status), parse_mode='HTML')

async def main():
    bot = TradingBot()
    try:
        await bot.start()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutting down gracefully...")
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
    finally:
        await bot.client.aclose()
        if bot.app:
            await bot.app.shutdown()

if __name__ == "__main__":
    # Verify required environment variables
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Error: Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID in environment")
        sys.exit(1)

    # Configure event loop policy for Windows compatibility
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    # Start the application
    asyncio.run(main())
