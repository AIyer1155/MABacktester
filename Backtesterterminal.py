import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt


#Settings#

Ticker = input("Enter TICKR:")
Start_date = "2020-01-01"
End_date = "2024-01-01"
FastWindow = 20
SlowWindow = 50


#Downloading the data

print(f"Downloading {Ticker} prices...")
raw = yf.download(Ticker, start=Start_date, end=End_date, progress=False)

if raw.empty:
    print(f"Error: '{Ticker}' not found. Check the ticker symbol and try again.")
    exit()

df = raw[["Close"]].copy()
df.columns = ["price"]
print(f"Got {len(df)} days of data \n")

#Moving Averages

df["fastavg"] = df["price"].rolling(window=FastWindow).mean()
df["slowavg"] = df["price"].rolling(window=SlowWindow).mean()

df = df.dropna()


#Generating signals
#Signal =1 means be in the market
#Signal =2 means be out of the market
#fastavg>slowavg means go in

df["signal"] = 0
df.loc[df["fastavg"] > df["slowavg"], "signal"]=1

#Returns

df["dailyreturn"] = df["price"].pct_change()

df["strategyreturn"] = df["dailyreturn"] * df["signal"].shift(1)


#Cummulative Performace
df["buy_and_hold"] = (1 + df["dailyreturn"]).cumprod()
df["strategy"]     = (1 + df["strategyreturn"]).cumprod()


#Output
bh_total  = df["buy_and_hold"].iloc[-1] - 1
str_total = df["strategy"].iloc[-1] - 1

print("═" * 40)
print(f"  Ticker:          {Ticker}")
print(f"  Period:          {Start_date} → {End_date}")
print(f"  Fast MA:         {FastWindow} days")
print(f"  Slow MA:         {SlowWindow} days")
print("─" * 40)
print(f"  Buy & hold:      {bh_total:+.1%}")
print(f"  MA strategy:     {str_total:+.1%}")
print(f"  Difference:      {str_total - bh_total:+.1%}")
print("═" * 40)

if str_total > bh_total:
    print("  The strategy beat buy-and-hold.")
else:
    print("  Buy-and-hold beat the strategy.")

#plot


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
fig.suptitle(f"{Ticker} — Moving Average Crossover ({FastWindow}/{SlowWindow} day)", fontsize=14)

# Top chart with the two moving averages and buy/sell markers
ax1.plot(df.index, df["price"],   label="Price",           color="gray",  linewidth=1)
ax1.plot(df.index, df["fastavg"], label=f"{FastWindow}d MA", color="blue",  linewidth=1.5)
ax1.plot(df.index, df["slowavg"], label=f"{SlowWindow}d MA", color="orange", linewidth=1.5)

buy_signals = df[(df["signal"] == 1) & (df["signal"].shift(1) == 0)]
ax1.scatter(buy_signals.index, buy_signals["price"], marker="^", color="green", s=80, label="Buy", zorder=5)

sell_signals = df[(df["signal"] == 0) & (df["signal"].shift(1) == 1)]
ax1.scatter(sell_signals.index, sell_signals["price"], marker="v", color="red", s=80, label="Sell", zorder=5)

ax1.set_ylabel("Price (USD)")
ax1.legend(loc="upper left")
ax1.grid(True, alpha=0.3)

# Bottom chart with cumulative returns comparison
ax2.plot(df.index, df["buy_and_hold"], label="Buy & hold", color="gray",  linewidth=1.5)
ax2.plot(df.index, df["strategy"],     label="MA strategy", color="blue",  linewidth=1.5)
ax2.axhline(y=1, color="black", linewidth=0.8, linestyle="--")
ax2.set_ylabel("Growth of $1")
ax2.set_xlabel("Date")
ax2.legend(loc="upper left")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{Ticker}_backtest.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"\nChart saved to {Ticker}_backtest.png")
