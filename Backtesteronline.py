import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="MABacktester", layout="centered")
st.title("MABacktester")

st.sidebar.header("Settings")
Ticker = st.sidebar.text_input("Enter TICKR:").upper()
Start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2020-01-01"))
End_date = st.sidebar.date_input("End date", value=pd.to_datetime("2024-01-01"))
FastWindow = 20
SlowWindow = 50
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

st.text("═" * 40)
st.text(f"  Ticker:          {Ticker}")
st.text(f"  Period:          {Start_date} → {End_date}")
st.text(f"  Fast MA:         {FastWindow} days")
st.text(f"  Slow MA:         {SlowWindow} days")
st.text(f"  Cost per trade:  {CostBps:.0f} bps")
st.text(f"  Trades:          {numtrades}")
st.text("─" * 40)
st.text(f"  Buy & hold:      {bh_total:+.1%}")
st.text(f"  MA strategy:     {str_total:+.1%}")
st.text(f"  MA net of cost:  {net_total:+.1%}")
st.text(f"  Difference:      {net_total - bh_total:+.1%}")
st.text("─" * 40)
st.text(f"  {'':16}{'Buy & hold':>12}{'Strategy':>12}")
st.text(f"  {'Sharpe':16}{bh_sharpe:>12.2f}{str_sharpe:>12.2f}")
st.text(f"  {'Volatility':16}{bh_vol:>12.1%}{str_vol:>12.1%}")
st.text(f"  {'Max drawdown':16}{bh_maxdd:>12.1%}{str_maxdd:>12.1%}")
st.text("═" * 40)

if net_total > bh_total:
    st.write("The strategy beat buy-and-hold after costs.")
else:
    st.write("Buy-and-hold beat the strategy after costs.")

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f"{Ticker} — Moving Average Crossover ({FastWindow}/{SlowWindow} day)", fontsize=14)

ax1.plot(df.index, df["price"], label="Price", color="gray", linewidth=1)
ax1.plot(df.index, df["fastavg"], label=f"{FastWindow}d MA", color="blue", linewidth=1.5)
ax1.plot(df.index, df["slowavg"], label=f"{SlowWindow}d MA", color="orange", linewidth=1.5)

buy_signals = df[(df["signal"] == 1) & (df["signal"].shift(1) == 0)]
ax1.scatter(buy_signals.index, buy_signals["price"], marker="^", color="green", s=80, label="Buy", zorder=5)

sell_signals = df[(df["signal"] == 0) & (df["signal"].shift(1) == 1)]
ax1.scatter(sell_signals.index, sell_signals["price"], marker="v", color="red", s=80, label="Sell", zorder=5)

ax1.set_ylabel("Price (USD)")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

ax2.plot(df.index, df["buy_and_hold"], label="Buy & hold", color="gray", linewidth=1.5)
ax2.plot(df.index, df["strategy_net"], label="MA strategy", color="blue", linewidth=1.5)
ax2.axhline(y=1, color="black", linewidth=0.8, linestyle="--")
ax2.set_ylabel("Growth of $1")
ax2.set_xlabel("Date")
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
st.pyplot(fig)
