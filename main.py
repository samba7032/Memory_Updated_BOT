import os
import time
import math
import requests
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

# =======================
# CONFIG
# =======================
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
CHAT_ID = os.getenv("CHAT_ID")

CSV_PATH = "under_100rs_stocks.csv"   # must contain a 'Symbol' column, e.g. ABC.NS
INTERVAL = "5m"                        # 1m/2m need premium; 5m is usually available
PERIOD = "5d"                          # enough history for EMA200 and ATR
LOOP_SECONDS = 60                      # check every minute
TZ_CHART = ZoneInfo("Asia/Kolkata")    # NSE time
SEND_PRICE_SYMBOL = "₹"                # currency sign for messages

# Strategy params (same as your Pine defaults)
FAST_LEN = 20
SLOW_LEN = 50
TREND_LEN = 200   # EMA
RSI_LEN = 14
RSI_OB = 70
RSI_OS = 30
ATR_LEN = 14
ATR_MULT_SL = 2.0
ATR_MULT_TP = 3.0
TRAIL_MULT = 1.5  # ATR trail in points

# =======================
# STATE
# =======================
last_bar_time = {}   # per symbol last processed bar (timestamp)
positions = {}       # per symbol: None or dict with position state

# position schema:
# {
#   'side': 'long' | 'short',
#   'entry_price': float,
#   'entry_time': pd.Timestamp,
#   'atr_at_entry': float,
#   'stop': float,
#   'take': float,
#   'trail_points': float,
#   'trail_stop': float,
# }

# =======================
# HELPERS
# =======================
def send_telegram(text: str):
    if not TELEGRAM_TOKEN or not CHAT_ID:
        print("⚠️ TELEGRAM_TOKEN or CHAT_ID not set; skipping send.")
        return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": CHAT_ID, "text": text})
        print(f"[{datetime.now(TZ_CHART).strftime('%H:%M:%S')}] 📲 {text.splitlines()[0]}")
    except Exception as e:
        print(f"❌ Telegram send failed: {e}")

def safe_download(symbol: str):
    """Download and sanity-check single-symbol data."""
    try:
        df = yf.download(symbol, period=PERIOD, interval=INTERVAL, auto_adjust=True, progress=False)
        if df is None or df.empty:
            return None
        req = {"Open","High","Low","Close","Volume"}
        if not req.issubset(df.columns):
            return None
        # Ensure 1D series per column
        for c in req:
            if getattr(df[c], "ndim", 1) != 1:
                return None
        return df
    except Exception:
        return None

def rsi_series(close: pd.Series, length: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def atr_series(high, low, close, length: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=length, min_periods=length).mean()
    return atr

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["SMA_FAST"] = out["Close"].rolling(FAST_LEN).mean()
    out["SMA_SLOW"] = out["Close"].rolling(SLOW_LEN).mean()
    out["EMA_TREND"] = out["Close"].ewm(span=TREND_LEN, adjust=False).mean()
    out["RSI"] = rsi_series(out["Close"], RSI_LEN)
    out["ATR"] = atr_series(out["High"], out["Low"], out["Close"], ATR_LEN)

    out["UPTREND"] = out["Close"] > out["EMA_TREND"]
    out["DOWNTREND"] = out["Close"] < out["EMA_TREND"]

    # Crossovers (same idea as Pine ta.crossover / crossunder)
    above = out["SMA_FAST"] > out["SMA_SLOW"]
    out["XUP"] = (above.astype(int).diff() == 1)
    out["XDN"] = (above.astype(int).diff() == -1)

    # Raw entry conditions
    out["RAW_BUY"]  = out["XUP"] & out["UPTREND"] & (out["RSI"] < RSI_OB)
    out["RAW_SELL"] = out["XDN"] & out["DOWNTREND"] & (out["RSI"] > RSI_OS)
    return out

def fmt_price(x: float) -> str:
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "NA"
    return f"{SEND_PRICE_SYMBOL}{x:.2f}"

# =======================
# ENTRY & EXIT LOGIC (mirrors your Pine)
# =======================
def try_entry(symbol: str, row_prev: pd.Series, row: pd.Series):
    """
    Use the **last fully closed bar** (row_prev) to trigger entries,
    and the **current forming bar** (row) for trailing/exit updates.
    """
    state = positions.get(symbol)
    in_long = (state is not None and state["side"] == "long")
    in_short = (state is not None and state["side"] == "short")

    # StrongBuy = rawBuy and not inTradeLong
    if row_prev["RAW_BUY"] and not in_long:
        entry_price = float(row_prev["Close"])
        atr = float(row_prev["ATR"])
        if math.isnan(atr):
            return  # not enough ATR history

        stop = entry_price - ATR_MULT_SL * atr
        take = entry_price + ATR_MULT_TP * atr
        trail_points = TRAIL_MULT * atr
        trail_stop = entry_price - trail_points  # initialize

        positions[symbol] = {
            "side": "long",
            "entry_price": entry_price,
            "entry_time": row_prev.name,
            "atr_at_entry": atr,
            "stop": stop,
            "take": take,
            "trail_points": trail_points,
            "trail_stop": trail_stop,
        }
        send_telegram(
            f"🚀 STRONG BUY\n"
            f"{symbol}\n"
            f"Entry: {fmt_price(entry_price)}  Time: {row_prev.name.tz_convert(TZ_CHART).strftime('%Y-%m-%d %H:%M')}\n"
            f"SL: {fmt_price(stop)}  TP: {fmt_price(take)}  Trail: {fmt_price(trail_points)}\n"
            f"RSI: {row_prev['RSI']:.2f}  Trend EMA200 OK"
        )
        return

    # StrongSell = rawSell and not inTradeShort
    if row_prev["RAW_SELL"] and not in_short:
        entry_price = float(row_prev["Close"])
        atr = float(row_prev["ATR"])
        if math.isnan(atr):
            return

        stop = entry_price + ATR_MULT_SL * atr
        take = entry_price - ATR_MULT_TP * atr
        trail_points = TRAIL_MULT * atr
        trail_stop = entry_price + trail_points

        positions[symbol] = {
            "side": "short",
            "entry_price": entry_price,
            "entry_time": row_prev.name,
            "atr_at_entry": atr,
            "stop": stop,
            "take": take,
            "trail_points": trail_points,
            "trail_stop": trail_stop,
        }
        send_telegram(
            f"🔻 STRONG SELL\n"
            f"{symbol}\n"
            f"Entry: {fmt_price(entry_price)}  Time: {row_prev.name.tz_convert(TZ_CHART).strftime('%Y-%m-%d %H:%M')}\n"
            f"SL: {fmt_price(stop)}  TP: {fmt_price(take)}  Trail: {fmt_price(trail_points)}\n"
            f"RSI: {row_prev['RSI']:.2f}  Trend EMA200 OK"
        )

def try_exit(symbol: str, row_prev: pd.Series, row: pd.Series):
    """
    Apply exits on the **current forming bar** (row) using Pine-like rules:
      - strategy.exit with stop, limit, trail_points
      - extra exit: RSI long<50 / short>50
    """
    state = positions.get(symbol)
    if not state:
        return

    side = state["side"]
    price_now = float(row["Close"])  # current bar price (forming)
    rsi_now = float(row["RSI"])

    # Update trailing stop (fixed trail_points from entry; move with favorable price)
    if side == "long":
        new_trail_stop = price_now - state["trail_points"]
        if new_trail_stop > state["trail_stop"]:
            state["trail_stop"] = new_trail_stop
    else:
        new_trail_stop = price_now + state["trail_points"]
        if new_trail_stop < state["trail_stop"]:
            state["trail_stop"] = new_trail_stop

    # Check exits in priority: hard SL/TP then trailing, then RSI rule
    exit_reason = None
    exit_price = price_now

    if side == "long":
        if price_now <= state["stop"]:
            exit_reason = "Stop Loss hit"
            exit_price = state["stop"]
        elif price_now >= state["take"]:
            exit_reason = "Take Profit hit"
            exit_price = state["take"]
        elif price_now <= state["trail_stop"]:
            exit_reason = "Trailing Stop hit"
            exit_price = state["trail_stop"]
        elif rsi_now < 50:
            exit_reason = "RSI < 50 exit"
    else:  # short
        if price_now >= state["stop"]:
            exit_reason = "Stop Loss hit"
            exit_price = state["stop"]
        elif price_now <= state["take"]:
            exit_reason = "Take Profit hit"
            exit_price = state["take"]
        elif price_now >= state["trail_stop"]:
            exit_reason = "Trailing Stop hit"
            exit_price = state["trail_stop"]
        elif rsi_now > 50:
            exit_reason = "RSI > 50 exit"

    if exit_reason:
        pnl = (exit_price - state["entry_price"]) * (1 if side == "long" else -1)
        send_telegram(
            f"✅ EXIT {side.upper()} • {symbol}\n"
            f"Reason: {exit_reason}\n"
            f"Entry: {fmt_price(state['entry_price'])} @ {state['entry_time'].tz_convert(TZ_CHART).strftime('%Y-%m-%d %H:%M')}\n"
            f"Exit : {fmt_price(exit_price)} @ {row.name.tz_convert(TZ_CHART).strftime('%Y-%m-%d %H:%M')}\n"
            f"PnL  : {fmt_price(pnl)}"
        )
        positions[symbol] = None

# =======================
# MAIN LOOP
# =======================
def load_symbols():
    try:
        df = pd.read_csv(CSV_PATH)
        syms = df["Symbol"].dropna().astype(str).unique().tolist()
        return syms
    except Exception as e:
        print(f"❌ Failed to load {CSV_PATH}: {e}")
        return []

def process_symbol(sym: str):
    df = safe_download(sym)
    if df is None or len(df) < max(TREND_LEN, SLOW_LEN, ATR_LEN) + 5:
        return  # not enough/clean data

    # yfinance indexes are tz-aware in UTC; convert for consistency
    if df.index.tz is None:
        df.index = df.index.tz_localize(timezone.utc)
    df = compute_indicators(df)

    # Work with the last two rows:
    # row_prev = last fully closed bar, row = current forming bar
    if len(df) < 3:
        return
    row_prev = df.iloc[-2]
    row = df.iloc[-1]

    # Avoid reprocessing same bar
    bar_time = row_prev.name
    if last_bar_time.get(sym) == bar_time:
        return
    last_bar_time[sym] = bar_time

    # ENTRY first (on closed bar), then EXIT updates on current bar
    try_entry(sym, row_prev, row)
    try_exit(sym, row_prev, row)

def main():
    syms = load_symbols()
    if not syms:
        print("❌ No symbols found. Ensure under_100rs_stocks.csv has a 'Symbol' column.")
        return

    for s in syms:
        positions[s] = None
        last_bar_time[s] = None

    send_telegram(f"✅ Python strategy started. Watching {len(syms)} symbols on {INTERVAL} bars.")

    while True:
        now = datetime.now(TZ_CHART).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{now}] Tick… checking {len(syms)} symbols")
        for s in syms:
            try:
                process_symbol(s)
            except Exception as e:
                # Keep running even if a symbol glitches
                print(f"❌ {s}: {e}")
        time.sleep(LOOP_SECONDS)

if __name__ == "__main__":
    main()
