import os
import yfinance as yf
import pandas as pd
import ta
import asyncio
import httpx
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes
from datetime import datetime, time as dtime, timedelta
from zoneinfo import ZoneInfo
from typing import Dict, List, Optional, Tuple
import logging
import gc
import psutil
from asyncio import Semaphore
import sys

# ===== Configuration =====
class Config:
    TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
    TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
    SYMBOLS_CSV = os.getenv('SYMBOLS_CSV', 'symbols.csv')
    DATA_DAYS = 5  # Reduced from 90 for memory efficiency
    CHECK_INTERVAL = 600  # 10 minutes
    RISK_REWARD_RATIO = 2
    MAX_CONCURRENT_REQUESTS = 5
    TEST_MODE = False
    STALE_DATA_THRESHOLD = 15
    MAX_RETRIES = 3
    BATCH_SIZE = 15

# ===== Logging Setup =====
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('bot.log')
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger('yfinance').setLevel(logging.WARNING)

class StockTradingBot:
    def __init__(self):
        self.app = None
        self.client = httpx.AsyncClient(timeout=30.0)
        self.signal_cache = {}
        self.market_timezone = ZoneInfo("Asia/Kolkata")
        self.symbols = []
        self.failed_symbols = set()
        self.api_semaphore = Semaphore(Config.MAX_CONCURRENT_REQUESTS)
        self.paused = False
        self.alert_queue = asyncio.Queue()
        self.sender_task = None

    async def initialize(self):
        """Initialize with memory-conscious loading"""
        try:
            self.symbols = await self.load_symbols()
            if not self.symbols:
                raise ValueError("No symbols loaded")
            
            self.app = Application.builder().token(Config.TELEGRAM_TOKEN).build()
            self.app.add_handler(CommandHandler("pause", self.pause))
            self.app.add_handler(CommandHandler("resume", self.resume))
            self.app.add_handler(CommandHandler("status", self.status))
            
            # Start background tasks
            self.sender_task = asyncio.create_task(self.alert_sender())
            await self.send_startup_message()
            return True
        except Exception as e:
            logger.critical(f"Initialization failed: {str(e)}")
            return False

    async def load_symbols(self) -> List[str]:
        """Memory-efficient symbol loading"""
        try:
            chunks = pd.read_csv(Config.SYMBOLS_CSV, chunksize=100)
            symbols = []
            for chunk in chunks:
                symbols.extend(chunk['Symbol'].dropna().str.strip().str.upper().tolist())
            return [f"{s.replace('.NS', '')}.NS" for s in symbols if isinstance(s, str)]
        except Exception as e:
            logger.error(f"Symbol loading failed: {str(e)}")
            return []

    async def run(self):
        """Robust main loop with memory management"""
        if not await self.initialize():
            return

        logger.info(f"==== {'TEST MODE' if Config.TEST_MODE else 'LIVE MODE'} ====")
        
        while True:
            try:
                if self.paused:
                    await asyncio.sleep(60)
                    continue

                if not self.is_market_open() and not Config.TEST_MODE:
                    await asyncio.sleep(Config.CHECK_INTERVAL)
                    continue

                start_time = asyncio.get_event_loop().time()
                await self.scan_symbols()
                
                # Dynamic sleep to maintain interval
                elapsed = asyncio.get_event_loop().time() - start_time
                sleep_time = max(0, Config.CHECK_INTERVAL - elapsed)
                await asyncio.sleep(sleep_time)
                
                # Force garbage collection
                gc.collect()
                
            except Exception as e:
                logger.error(f"Main loop error: {str(e)}")
                await asyncio.sleep(60)

    async def scan_symbols(self):
        """Efficient symbol scanning with memory limits"""
        active_symbols = [s for s in self.symbols if s not in self.failed_symbols]
        
        for i in range(0, len(active_symbols), Config.BATCH_SIZE):
            batch = active_symbols[i:i + Config.BATCH_SIZE]
            batch_data = await self.fetch_batch_data(batch)
            
            for symbol, data in batch_data.items():
                if data is not None:
                    signal, price, notes = self.generate_signal(symbol, data)
                    if signal in ('BUY', 'SELL'):
                        await self.queue_alert(signal, symbol, price, notes)
            
            # Clear memory between batches
            del batch_data
            gc.collect()

    async def fetch_batch_data(self, symbols: List[str]) -> Dict[str, Optional[pd.DataFrame]]:
        """Throttled batch fetching with retries"""
        results = {}
        for attempt in range(Config.MAX_RETRIES):
            failed = []
            async with self.api_semaphore:
                tasks = {symbol: self.fetch_stock_data(symbol) for symbol in symbols}
                completed = await asyncio.gather(*tasks.values(), return_exceptions=True)
                
                for symbol, result in zip(tasks.keys(), completed):
                    if isinstance(result, Exception):
                        failed.append(symbol)
                    else:
                        results[symbol] = result
            
            if not failed or attempt == Config.MAX_RETRIES - 1:
                break
                
            symbols = failed
            await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return results

    async def fetch_stock_data(self, symbol: str) -> Optional[pd.DataFrame]:
        """Memory-efficient data fetching"""
        try:
            period = "1d" if self.is_market_open() else f"{Config.DATA_DAYS}d"
            interval = "15m" if self.is_market_open() else "1d"
            
            data = await asyncio.to_thread(
                yf.download,
                symbol,
                period=period,
                interval=interval,
                progress=False,
                auto_adjust=True,
                threads=False  # Reduces memory usage
            )
            return data if not data.empty else None
        except Exception as e:
            logger.debug(f"Data fetch failed for {symbol}: {str(e)}")
            return None

    # [Rest of your methods (generate_signal, is_market_open, etc.) remain similar]
    # Add the improved alert_sender and queue_alert methods from previous examples

async def main():
    bot = StockTradingBot()
    try:
        await bot.run()
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        await bot.shutdown()

if __name__ == "__main__":
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    try:
        asyncio.run(main())
    except Exception as e:
        logger.critical(f"Fatal error: {str(e)}")
