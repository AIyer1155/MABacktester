# MABacktester

A moving average crossover backtester built in Python. Enter any stock ticker, pick a date range and MA windows, and instantly see whether the strategy would have beaten buy-and-hold with an interactive web dashboard.



---

## What it does

- Pulls historical price data for any ticker via **yfinance**
- Computes a fast and slow moving average using **pandas**
- Generates buy/sell signals when the MAs cross
- Calculates cumulative returns and compares them against buy-and-hold
- Displays results in an interactive **Streamlit** web dashboard

## Quick start

```bash
git clone https://github.com/aiyer115/MABacktester.git
cd MABacktester
pip install yfinance pandas matplotlib streamlit
streamlit run Backtesteronline.py
```

## How it works

When the fast MA crosses **above** the slow MA, go long. When it crosses **below**, exit to cash.

```
signal = 1  →  hold the stock   (fast MA > slow MA)
signal = 0  →  hold cash        (fast MA < slow MA)
```

Daily returns are only captured on days when `signal = 1`. Cumulative returns are computed as:

```
cumulative = (1 + r₁) × (1 + r₂) × ... × (1 + rₙ)
```

The dashboard plots both the strategy and buy-and-hold so you can see exactly where the signals added or lost value.

## Example output

| Ticker | Period | Fast MA | Slow MA | Buy & Hold | Strategy |
|--------|--------|---------|---------|------------|----------|
| AAPL   | 2020–2024 | 20d | 50d | +148% | +91% |
| ^NSEI  | 2020–2024 | 20d | 50d | +112% | +67% |


## Stack

| Tool | Purpose |
|------|---------|
| `yfinance` | Price data from Yahoo Finance |
| `pandas` | Data manipulation and MA calculation |
| `matplotlib` | Chart rendering |
| `streamlit` | Web dashboard |

## What I learned

This project was built to understand how quantitative strategies are constructed and evaluated from scratch, specifically how signal generation, position sizing, and return attribution work before adding complexity like transaction costs or risk metrics.

## Roadmap

- [ ] Add transaction cost modeling
- [ ] Support multiple tickers simultaneously  
- [ ] Add Sharpe ratio and max drawdown metrics
- [ ] Export results to CSV

---

Built by Advay Iyer www.linkedin.com/in/advay-iyer-520506339

