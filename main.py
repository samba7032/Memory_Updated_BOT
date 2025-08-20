import yfinance as yf
import pandas as pd
import ta
import time
import requests
from datetime import datetime

# === TELEGRAM SETUP ===
TELEGRAM_TOKEN = "8294290613:AAHTFv8g65vKkTYwWM2urX0XM9vPEa0oR64"
TELEGRAM_CHAT_ID = "1020815701"

def send_telegram(msg: str):
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print("Telegram send failed:", e)

# === STRATEGY IMPLEMENTATION (same as PineScript) ===
def strategy_signals(df):
    if df is None or len(df) < 200:
        return None

    # Indicators
    df["fastMA"] = df["Close"].rolling(20).mean()
    df["slowMA"] = df["Close"].rolling(50).mean()
    df["trendEMA"] = df["Close"].ewm(span=200).mean()
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"], 14).rsi()
    df["atr"] = ta.volatility.AverageTrueRange(
        df["High"], df["Low"], df["Close"], 14
    ).average_true_range()

    # Trend
    df["uptrend"] = df["Close"] > df["trendEMA"]
    df["downtrend"] = df["Close"] < df["trendEMA"]

    # Crossovers
    df["rawBuy"] = (df["fastMA"] > df["slowMA"]) & (df["fastMA"].shift(1) <= df["slowMA"].shift(1)) & df["uptrend"] & (df["rsi"] < 70)
    df["rawSell"] = (df["fastMA"] < df["slowMA"]) & (df["fastMA"].shift(1) >= df["slowMA"].shift(1)) & df["downtrend"] & (df["rsi"] > 30)

    latest = df.iloc[-1]

    if latest["rawBuy"]:
        return f"📈 STRONG BUY {latest.name} @ {latest['Close']:.2f}"
    elif latest["rawSell"]:
        return f"📉 STRONG SELL {latest.name} @ {latest['Close']:.2f}"
    else:
        return None

# === MAIN LOOP ===
def run_monitor(stock_list):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] 📲 ✅ Python strategy started. Watching {len(stock_list)} symbols on 5m bars.")
    while True:
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Tick… checking {len(stock_list)} symbols")
        for symbol in stock_list:
            try:
                # Ensure NSE suffix
                ticker = symbol if symbol.endswith(".NS") else symbol + ".NS"
                df = yf.download(ticker, period="5d", interval="5m", progress=False)

                if df is None or df.empty:
                    print(f"⚠️ No data for {symbol}")
                    continue

                signal = strategy_signals(df)
                if signal:
                    send_telegram(f"{signal} ({symbol})")

            except Exception as e:
                print(f"❌ Error processing {symbol}: {e}")

        time.sleep(300)  # wait 5 minutes

# === LOAD STOCK LIST ===
stock_list = pd.read_csv("under_100rs_stocks.csv")["Symbol"].tolist()
run_monitor(stock_list)
