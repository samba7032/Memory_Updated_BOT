import pandas as pd
import yfinance as yf
import talib as ta
import time
import requests
from datetime import datetime
import pytz

# === SETTINGS ===
csv_file = "under_200_to_400_stocks.csv"     # your CSV file with "Symbols" column
interval = "5m"             # timeframe (1m, 5m, 15m etc.)
lookback = "5d"             # how much history to load

# Market hours (India)
IST = pytz.timezone("Asia/Kolkata")
market_open = datetime.strptime("09:15", "%H:%M").time()
market_close = datetime.strptime("15:30", "%H:%M").time()

# Telegram Bot (optional) - leave blank if not needed
TELEGRAM_TOKEN = "8418510043:AAELHmMILdUZ2Mn80KA7ymMKJSysYgHV5aI"   # e.g. "123456:ABC-DEF..."
CHAT_ID = "1020815701"          # your Telegram chat ID

def send_telegram(message):
    """Send alert to Telegram (if configured) + print to console"""
    if TELEGRAM_TOKEN and CHAT_ID:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        try:
            requests.post(url, data={"chat_id": CHAT_ID, "text": message})
        except Exception as e:
            print(f"⚠️ Telegram error: {e}")
    print(message)

def fetch_data(symbol):
    """Download stock OHLCV data"""
    df = yf.download(
        symbol,
        period=lookback,
        interval=interval,
        progress=False,
        auto_adjust=False   # fixes FutureWarning
    )
    df.dropna(inplace=True)
    return df

def generate_signals(df):
    # Ensure enough data
    if len(df) < 200:
        return df

    close = df["Close"].astype(float).values

    # Check 1D array
    if close.ndim != 1:
        return df

    # Drop NaNs
    close = close[~pd.isna(close)]

    # Still enough after removing NaNs?
    if len(close) < 200:
        return df

    df["fastMA"] = ta.SMA(close, timeperiod=20)
    df["slowMA"] = ta.SMA(close, timeperiod=50)
    df["trendEMA"] = ta.EMA(close, timeperiod=200)
    df["RSI"] = ta.RSI(close, timeperiod=14)

    df["Uptrend"] = df["Close"] > df["trendEMA"]
    df["Downtrend"] = df["Close"] < df["trendEMA"]

    df["BuySignal"] = (
        (df["fastMA"].shift(1) < df["slowMA"].shift(1)) &
        (df["fastMA"] > df["slowMA"]) &
        df["Uptrend"] &
        (df["RSI"] < 70)
    )

    df["SellSignal"] = (
        (df["fastMA"].shift(1) > df["slowMA"].shift(1)) &
        (df["fastMA"] < df["slowMA"]) &
        df["Downtrend"] &
        (df["RSI"] > 30)
    )

    return df

def run():
    # read stock list
    stocks = pd.read_csv(csv_file)["Symbols"].dropna().tolist()
    print(f"Monitoring {len(stocks)} stocks...")

    # Track trade states per stock
    trade_state = {s: {"long": False, "short": False, "last_signal": None} for s in stocks}

    while True:
        now = datetime.now(IST).time()

        if now < market_open or now > market_close:
            print("⏸ Market Closed... waiting for next session")
            time.sleep(60)
            continue

        for symbol in stocks:
            try:
                df = fetch_data(symbol)
                df = generate_signals(df)

                # Skip stocks with insufficient data
                if len(df) < 200:
                    continue

                latest = df.iloc[-1]
                state = trade_state[symbol]

                # only check on new candle
                if latest.name != state["last_signal"]:
                    if latest.get("BuySignal", False) and not state["long"]:
                        msg = f"🚀 BUY {symbol} at {latest['Close']:.2f}"
                        send_telegram(msg)
                        state["long"], state["short"] = True, False
                        state["last_signal"] = latest.name

                    elif latest.get("SellSignal", False) and not state["short"]:
                        msg = f"🔻 SELL {symbol} at {latest['Close']:.2f}"
                        send_telegram(msg)
                        state["short"], state["long"] = True, False
                        state["last_signal"] = latest.name

            except Exception as e:
                print(f"⚠️ Error for {symbol}: {e}")

        time.sleep(60)  # wait 1 min

if __name__ == "__main__":
    run()

