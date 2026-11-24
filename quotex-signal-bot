import numpy as np
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Quotex Live Signal Assistant", page_icon="📊", layout="centered")

PAIRS = [
"EUR/USD",
"EUR/JPY",
"GBP/USD",
"USD/JPY",
"AUD/USD",
"USD/CAD",
"GBP/JPY",
"AUD/JPY",
"NZD/USD",
"USD/CHF",
]


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
delta = series.diff()
gain = delta.clip(lower=0)
loss = -delta.clip(upper=0)
avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
rs = avg_gain / avg_loss.replace(0, np.nan)
rsi = 100 - (100 / (1 + rs))
return rsi.fillna(50)


def fetch_live_candles(pair: str, api_key: str, interval: str = "1min", outputsize: int = 60):
url = "https://api.twelvedata.com/time_series"
params = {
"symbol": pair,
"interval": interval,
"outputsize": outputsize,
"apikey": api_key.strip(),
}
resp = requests.get(url, params=params, timeout=15)
data = resp.json()

if resp.status_code != 200 or data.get("status") == "error":
msg = data.get("message", "Unable to fetch live market data.")
raise RuntimeError(msg)

values = data.get("values")
if not values or len(values) < 25:
raise RuntimeError("Not enough real-time candle data returned yet.")

df = pd.DataFrame(values)
df["datetime"] = pd.to_datetime(df["datetime"])
for col in ["open", "high", "low", "close"]:
df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["open", "high", "low", "close"]).sort_values("datetime").reset_index(drop=True)
if len(df) < 25:
raise RuntimeError("Insufficient valid candles after cleaning.")
return df


def analyze_market(df: pd.DataFrame) -> dict:
close = df["close"]
open_p = df["open"]
high_p = df["high"]
low_p = df["low"]

ema9 = close.ewm(span=9, adjust=False).mean()
ema21 = close.ewm(span=21, adjust=False).mean()
rsi = compute_rsi(close, 14)

last_close = float(close.iloc[-1])
last_open = float(open_p.iloc[-1])
last_ema9 = float(ema9.iloc[-1])
last_ema21 = float(ema21.iloc[-1])
prev_ema9 = float(ema9.iloc[-2])
prev_ema21 = float(ema21.iloc[-2])
last_rsi = float(rsi.iloc[-1])

recent_low = float(low_p.iloc[-15:-1].min())
recent_high = float(high_p.iloc[-15:-1].max())

bull_score = 0
bear_score = 0
reasons = []

# 1. Trend & EMA 9 / EMA 21 alignment
if last_ema9 > last_ema21 and last_close > last_ema9:
bull_score += 35
reasons.append("EMA 9 > EMA 21 & Price above EMA 9 (Bullish bias)")
elif last_ema9 < last_ema21 and last_close < last_ema9:
bear_score += 35
reasons.append("EMA 9 < EMA 21 & Price below EMA 9 (Bearish bias)")
else:
reasons.append("EMA 9 / EMA 21 mixed or sideways")

# Fresh crossover bonus
if prev_ema9 <= prev_ema21 and last_ema9 > last_ema21:
bull_score += 15
reasons.append("Fresh bullish EMA 9/21 crossover")
elif prev_ema9 >= prev_ema21 and last_ema9 < last_ema21:
bear_score += 15
reasons.append("Fresh bearish EMA 9/21 crossover")

# 2. RSI momentum
if 52 <= last_rsi <= 72:
bull_score += 25
reasons.append(f"RSI ({last_rsi:.1f}) shows bullish momentum")
elif 28 <= last_rsi <= 48:
bear_score += 25
reasons.append(f"RSI ({last_rsi:.1f}) shows bearish momentum")
else:
reasons.append(f"RSI ({last_rsi:.1f}) is neutral or extreme")

# 3. Last candle direction
if last_close > last_open:
bull_score += 15
reasons.append("Latest candle is bullish (green)")
elif last_close < last_open:
bear_score += 15
reasons.append("Latest candle is bearish (red)")
else:
reasons.append("Latest candle is doji/flat")

# 4. Proximity to recent support / resistance
range_span = max(recent_high - recent_low, 1e-9)
near_support = abs(last_close - recent_low) / range_span <= 0.18
near_resistance = abs(recent_high - last_close) / range_span <= 0.18

if near_support and last_close >= last_open:
bull_score += 10
reasons.append("Bouncing near recent support zone")
if near_resistance and last_close <= last_open:
bear_score += 10
reasons.append("Rejecting near recent resistance zone")

max_score = max(bull_score, bear_score)

# Strict threshold: below 60% -> NO TRADE
if max_score < 60:
confidence = max(25, min(59, max_score))
return {
"signal": "NO TRADE",
"confidence": confidence,
"price": last_close,
"ema9": last_ema9,
"ema21": last_ema21,
"rsi": last_rsi,
"support": recent_low,
"resistance": recent_high,
"reasons": reasons + ["Market conditions are weak or mixed — avoid entry"],
}

if bull_score > bear_score:
return {
"signal": "BUY (CALL)",
"confidence": min(95, bull_score),
"price": last_close,
"ema9": last_ema9,
"ema21": last_ema21,
"rsi": last_rsi,
"support": recent_low,
"resistance": recent_high,
"reasons": reasons,
}
else:
return {
"signal": "SELL (PUT)",
"confidence": min(95, bear_score),
"price": last_close,
"ema9": last_ema9,
"ema21": last_ema21,
"rsi": last_rsi,
"support": recent_low,
"resistance": recent_high,
"reasons": reasons,
}


st.title("📊 Quotex Live Market Signal Assistant")
st.caption("Manual entry only • Verified live market data • No profit guarantee")

with st.expander("🔑 Twelve Data API Key", expanded=True):
api_key = st.text_input("Enter your Twelve Data API Key", type="password", placeholder="Paste your API key here")

selected_pair = st.selectbox("Select Trading Pair (10 Pairs)", PAIRS, index=1)
timeframe = st.selectbox("Candle Interval", ["1min", "5min"], index=0)

if st.button("▶ Start Analysis", type="primary", use_container_width=True):
if not api_key.strip():
st.warning("Please enter your Twelve Data API Key first.")
else:
with st.spinner(f"Fetching live market data for {selected_pair}..."):
try:
df = fetch_live_candles(selected_pair, api_key, interval=timeframe)
result = analyze_market(df)

st.divider()
st.subheader(f"{selected_pair} ({timeframe}) — {result['signal']}")
st.metric("Confidence Score", f"{result['confidence']}%")
st.metric("Latest Price", f"{result['price']:.5f}")

col1, col2, col3 = st.columns(3)
col1.metric("EMA 9", f"{result['ema9']:.5f}")
col2.metric("EMA 21", f"{result['ema21']:.5f}")
col3.metric("RSI (14)", f"{result['rsi']:.1f}")

st.write(f"**Support:** `{result['support']:.5f}` | **Resistance:** `{result['resistance']:.5f}`")
st.write("**Analysis Details:**")
for item in result["reasons"]:
st.write(f"- {item}")

st.info("Reminder: This is a technical analysis assistant for manual entry. Test on Demo first.")
except Exception as err:
st.error(f"Live data error: {err}")
