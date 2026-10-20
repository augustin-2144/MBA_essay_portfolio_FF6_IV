# Handles data downloading, cleaning, and return computation
import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf

# Download price data
def download_prices(tickers, start, end, interval="1mo"):
    """
    Downloads price data using yfinance.
    """

    prices = yf.download(
        tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True
    )["Close"]
    prices = prices.dropna(how="all")
    return prices

# Frozen price snapshot (load if present, else fetch live and freeze)
def load_or_fetch_prices(tickers, start, end, snapshot_path, interval="1mo"):
    """
    Loads a frozen price snapshot from `snapshot_path` if it exists, otherwise
    downloads prices live via `download_prices` and saves them to `snapshot_path`
    for every later run to reuse.

    Even with `start`/`end` held fixed, yfinance's adjusted-close series is not
    perfectly stable over time (dividend/split adjustment factors get revised
    after the fact), so without freezing, re-running the notebook weeks apart
    can silently shift historical returns, Sigma_hist, and every downstream
    result. Delete `snapshot_path` to force a fresh live pull and re-freeze on
    a new date.
    """

    if snapshot_path.exists():
        prices = pd.read_csv(snapshot_path, comment="#", index_col=0, parse_dates=True)
        with open(snapshot_path) as f:
            header = f.readline()
        print(f"Prices loaded from frozen snapshot ({header.lstrip('# ').strip()})")
        return prices[tickers] if len(tickers) > 1 else prices

    prices = download_prices(tickers=tickers, start=start, end=end, interval=interval)

    snapshot_date = dt.date.today().isoformat()
    with open(snapshot_path, "w") as f:
        f.write(f"# Price snapshot retrieved on {snapshot_date} (start={start}, end={end})\n")
        prices.to_csv(f)

    print(f"Prices fetched live and frozen to snapshot: {snapshot_date}")
    return prices

# Compute returns
def compute_returns(prices, method="simple"):
    """
    Computes returns from price data.

    Parameters
    ----------
    method : str
        "simple" or "log"
    """

    if method == "log":
        returns = np.log(prices / prices.shift(1))
    else:
        returns = prices.pct_change()
    return returns.dropna()

# Align data
def align_data(returns, factors):
    """
    Aligns asset returns with factor data.
    """

    returns.index = pd.to_datetime(returns.index).to_period("M")
    data = returns.join(factors, how="inner")
    return data