import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Moving Average Backtester", layout="centered")
st.markdown("""
<style>
.block-container { padding-top: 2rem; }
</style>
<div style="border: 1px solid #2a2f3a; border-radius: 8px; padding: 12px 20px; margin-bottom: 16px; background-color: #161a23;">
  <h1 style="margin: 0; color: white;">Moving Average Backtester</h1>
</div>
""", unsafe_allow_html=True)

st.sidebar.header("Settings")
Ticker = st.sidebar.text_input("Enter TICKR:").upper()
Ticker2 = st.sidebar.text_input("Compare with (optional):").upper()
Start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2020-01-01"))
End_date = st.sidebar.date_input("End date", value=pd.to_datetime("2024-01-01"))
FastWindow = st.sidebar.slider("Fast MA (days)", min_value=5, max_value=100, value=20)
SlowWindow = st.sidebar.slider("Slow MA (days)", min_value=10, max_value=300, value=50)
CostBps = st.sidebar.number_input("Cost per trade (bps)", value=10.0)

run = st.sidebar.button("Run", type="primary")

if not run:
    st.info("Configure your settings in the sidebar and click Run.")
    st.stop()

raw = yf.download(Ticker, start=Start_date, end=End_date, progress=False)

if raw.empty:
    st.error(f"Error: '{Ticker}' not found. Check the ticker symbol and try again.")
    st.stop()

df = raw[["Close"]].copy()
df.columns = ["price"]
st.write(f"Got {len(df)} days of data")

df["fastavg"] = df["price"].rolling(window=FastWindow).mean()
df["slowavg"] = df["price"].rolling(window=SlowWindow).mean()

df = df.dropna()

df["signal"] = 0
df.loc[df["fastavg"] > df["slowavg"], "signal"] = 1

df["dailyreturn"] = df["price"].pct_change()

df["strategyreturn"] = df["dailyreturn"] * df["signal"].shift(1)

df["trade"] = df["signal"].diff().abs().fillna(0)
df["cost"] = df["trade"] * (CostBps / 10000)
df["strategyreturn_net"] = df["strategyreturn"] - df["cost"]

df["buy_and_hold"] = (1 + df["dailyreturn"]).cumprod()
df["strategy"] = (1 + df["strategyreturn"]).cumprod()
df["strategy_net"] = (1 + df["strategyreturn_net"]).cumprod()

bh_total = df["buy_and_hold"].iloc[-1] - 1
str_total = df["strategy"].iloc[-1] - 1
net_total = df["strategy_net"].iloc[-1] - 1

bh_sharpe = df["dailyreturn"].mean() / df["dailyreturn"].std() * np.sqrt(252)
str_sharpe = df["strategyreturn_net"].mean() / df["strategyreturn_net"].std() * np.sqrt(252)

bh_vol = df["dailyreturn"].std() * np.sqrt(252)
str_vol = df["strategyreturn_net"].std() * np.sqrt(252)

bh_peak = df["buy_and_hold"].cummax()
str_peak = df["strategy_net"].cummax()
bh_maxdd = ((df["buy_and_hold"] - bh_peak) / bh_peak).min()
str_maxdd = ((df["strategy_net"] - str_peak) / str_peak).min()

numtrades = int(df["trade"].sum())

st.subheader(f"{Ticker}  ·  {FastWindow}/{SlowWindow} day  ·  {numtrades} trades  ·  {CostBps:.0f} bps")

if net_total > bh_total:
    st.success(f"The strategy beat buy-and-hold after costs by {net_total - bh_total:+.1%}.")
else:
    st.error(f"Buy-and-hold beat the strategy after costs by {bh_total - net_total:.1%}.")

col1, col2, col3 = st.columns(3)
col1.metric("Buy & hold", f"{bh_total:+.1%}")
col2.metric("MA strategy (net)", f"{net_total:+.1%}", f"{net_total - bh_total:+.1%}")
col3.metric("Gross of cost", f"{str_total:+.1%}")

col4, col5, col6 = st.columns(3)
col4.metric("Sharpe", f"{str_sharpe:.2f}", f"{str_sharpe - bh_sharpe:+.2f} vs B&H")
col5.metric("Volatility", f"{str_vol:.1%}")
col6.metric("Max drawdown", f"{str_maxdd:.1%}", f"{str_maxdd - bh_maxdd:+.1%} vs B&H")

if Ticker2:
    raw2 = yf.download(Ticker2, start=Start_date, end=End_date, progress=False)
    if raw2.empty:
        st.warning(f"'{Ticker2}' not found, skipping comparison.")
        df2 = None
    else:
        df2 = raw2[["Close"]].copy()
        df2.columns = ["price"]
        df2["fastavg"] = df2["price"].rolling(window=FastWindow).mean()
        df2["slowavg"] = df2["price"].rolling(window=SlowWindow).mean()
        df2 = df2.dropna()
        df2["signal"] = 0
        df2.loc[df2["fastavg"] > df2["slowavg"], "signal"] = 1
        df2["dailyreturn"] = df2["price"].pct_change()
        df2["strategyreturn"] = df2["dailyreturn"] * df2["signal"].shift(1)
        df2["trade"] = df2["signal"].diff().abs().fillna(0)
        df2["cost"] = df2["trade"] * (CostBps / 10000)
        df2["strategyreturn_net"] = df2["strategyreturn"] - df2["cost"]
        df2["strategy_net"] = (1 + df2["strategyreturn_net"]).cumprod()
else:
    df2 = None

plt.style.use("dark_background")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.patch.set_facecolor("#0E1117")
ax1.set_facecolor("#0E1117")
ax2.set_facecolor("#0E1117")
fig.suptitle(f"{Ticker} — Moving Average Crossover ({FastWindow}/{SlowWindow} day)", fontsize=14, color="white")

ax1.plot(df.index, df["price"], label="Price", color="lightgray", linewidth=1)
ax1.plot(df.index, df["fastavg"], label=f"{FastWindow}d MA", color="deepskyblue", linewidth=1.5)
ax1.plot(df.index, df["slowavg"], label=f"{SlowWindow}d MA", color="orange", linewidth=1.5)

buy_signals = df[(df["signal"] == 1) & (df["signal"].shift(1) == 0)]
ax1.scatter(buy_signals.index, buy_signals["price"], marker="^", color="limegreen", s=80, label="Buy", zorder=5)

sell_signals = df[(df["signal"] == 0) & (df["signal"].shift(1) == 1)]
ax1.scatter(sell_signals.index, sell_signals["price"], marker="v", color="red", s=80, label="Sell", zorder=5)

ax1.set_ylabel("Price (USD)")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.2)

ax2.plot(df.index, df["buy_and_hold"], label=f"{Ticker} buy & hold", color="lightgray", linewidth=1.5)
ax2.plot(df.index, df["strategy_net"], label=f"{Ticker} MA strategy", color="deepskyblue", linewidth=1.5)
if df2 is not None:
    ax2.plot(df2.index, df2["strategy_net"], label=f"{Ticker2} MA strategy", color="violet", linewidth=1.5)
ax2.axhline(y=1, color="white", linewidth=0.8, linestyle="--")
ax2.set_ylabel("Growth of $1")
ax2.set_xlabel("Date")
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.2)

plt.tight_layout()
st.pyplot(fig)
