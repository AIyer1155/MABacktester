import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="MABacktester", layout="centered")
st.title("MABacktester")

st.sidebar.header("Settings")
Ticker = st.sidebar.text_input("Enter TICKR:", value="AAPL").upper()
Start_date = st.sidebar.date_input("Start date", value=pd.to_datetime("2020-01-01"))
End_date = st.sidebar.date_input("End date", value=pd.to_datetime("2024-01-01"))
FastWindow = st.sidebar.slider("Fast MA (days)", min_value=5, max_value=100, value=20)
SlowWindow = st.sidebar.slider("Slow MA (days)", min_value=10, max_value=300, value=50)

run = st.sidebar.button("Run", type="primary")

if not run:
    st.info("Configure your settings in the sidebar and click Run.")
    st.stop()

@st.cache_data(ttl=3600)
def get_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end, progress=False)

raw = get_data(Ticker, Start_date, End_date)

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

df["buy_and_hold"] = (1 + df["dailyreturn"]).cumprod()
df["strategy"] = (1 + df["strategyreturn"]).cumprod()

bh_total  = df["buy_and_hold"].iloc[-1] - 1
str_total = df["strategy"].iloc[-1] - 1
diff      = str_total - bh_total

# Metric cards
col1, col2, col3 = st.columns(3)
col1.metric("Buy & hold",  f"{bh_total:+.1%}")
col2.metric("MA strategy", f"{str_total:+.1%}")
col3.metric("Difference",  f"{diff:+.1%}", delta=f"{diff:+.1%}")

# Verdict banner
if str_total > bh_total:
    st.success("The strategy beat buy-and-hold.")
else:
    st.warning("Buy-and-hold beat the strategy.")

# Dark-themed chart
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f"{Ticker} — Moving Average Crossover ({FastWindow}/{SlowWindow} day)", fontsize=14, color="white")
fig.patch.set_facecolor("#0e1117")
for ax in (ax1, ax2):
    ax.set_facecolor("#0e1117")
    ax.tick_params(colors="white")
    ax.yaxis.label.set_color("white")
    ax.xaxis.label.set_color("white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#333")

ax1.plot(df.index, df["price"],   label="Price",              color="#555",    linewidth=1)
ax1.plot(df.index, df["fastavg"], label=f"{FastWindow}d MA",  color="#3b82f6", linewidth=1.5)
ax1.plot(df.index, df["slowavg"], label=f"{SlowWindow}d MA",  color="#f59e0b", linewidth=1.5)

buy_signals  = df[(df["signal"] == 1) & (df["signal"].shift(1) == 0)]
sell_signals = df[(df["signal"] == 0) & (df["signal"].shift(1) == 1)]
ax1.scatter(buy_signals.index,  buy_signals["price"],  marker="^", color="#10b981", s=80, zorder=5, label="Buy")
ax1.scatter(sell_signals.index, sell_signals["price"], marker="v", color="#ef4444", s=80, zorder=5, label="Sell")

ax1.set_ylabel("Price (USD)", color="white")
ax1.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
ax1.grid(True, alpha=0.1)

ax2.plot(df.index, df["buy_and_hold"], label="Buy & hold",  color="#888",    linewidth=1.5)
ax2.plot(df.index, df["strategy"],     label="MA strategy", color="#3b82f6", linewidth=1.5)
ax2.axhline(y=1, color="white", linewidth=0.8, linestyle="--", alpha=0.3)
ax2.set_ylabel("Growth of $1", color="white")
ax2.set_xlabel("Date", color="white")
ax2.legend(facecolor="#1a1a2e", labelcolor="white", fontsize=8)
ax2.grid(True, alpha=0.1)

plt.tight_layout()
st.pyplot(fig)
