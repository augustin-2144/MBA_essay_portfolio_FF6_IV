import numpy as np
import pandas as pd
import yfinance as yf
import datetime as dt
from black_scholes import implied_volatility_call


# Maturity selection
def choose_maturity(maturities, target_days=30):
    """
    Selects the option maturity closest to a target number of calendar days.
    """

    today = dt.datetime.today()
    best_maturity = None
    min_diff = float("inf")

    for maturity in maturities:
        maturity_date = dt.datetime.strptime(maturity, "%Y-%m-%d")
        days_to_maturity = (maturity_date - today).days

        if days_to_maturity <= 0:
            continue

        diff = abs(days_to_maturity - target_days)

        if diff < min_diff:
            min_diff = diff
            best_maturity = maturity

    return best_maturity


# Dividend yield lookup
def get_dividend_yield(tk):
    """
    Reads the trailing dividend yield for a yfinance Ticker, normalized to a
    continuous annual rate (e.g. 0.024 for 2.4%).

    `info["dividendYield"]` is returned by the current yfinance version as a
    percentage number (e.g. 2.4 for 2.4%), not a fraction - a unit that has
    changed across library versions in the past. Missing data (None) or an
    implausible value (>25%, almost certainly a stale unit convention rather
    than a real yield) both default to 0, i.e. the previous no-dividend
    behavior, rather than risk silently corrupting the extraction.
    """

    dy = tk.info.get("dividendYield")
    if dy is None:
        return 0.0
    q = dy / 100
    return q if q <= 0.25 else 0.0


# ATM implied volatility for one ticker
def get_atm_implied_vol(ticker, r, target_days=30):
    """
    Computes the at-the-money implied volatility for one ticker.
    """

    try:
        tk = yf.Ticker(ticker)

        # 1. Spot price
        history = tk.history(period="1d")
        if history.empty:
            return np.nan
        S = history["Close"].iloc[-1]

        # 2. Maturity selection
        maturities = tk.options
        if len(maturities) == 0:
            return np.nan
        maturity = choose_maturity(maturities, target_days=target_days)
        if maturity is None:
            return np.nan

        # 3. Option chain
        chain = tk.option_chain(maturity)
        calls = chain.calls.copy()
        calls = calls[calls["lastPrice"] > 0]
        if calls.empty:
            return np.nan

        # 4. ATM call selection
        calls["distance"] = abs(calls["strike"] - S)
        calls = calls.sort_values("distance")
        atm_call = calls.iloc[0]
        K = atm_call["strike"]
        C_market = atm_call["lastPrice"]

        # 5. Time to maturity
        today = dt.datetime.today()
        maturity_date = dt.datetime.strptime(maturity, "%Y-%m-%d")
        T = (maturity_date - today).days / 365
        if T <= 0:
            return np.nan

        # 6. Dividend yield
        q = get_dividend_yield(tk)

        # 7. Black-Scholes-Merton inversion
        sigma = implied_volatility_call(
            C_market=C_market,
            S=S,
            K=K,
            T=T,
            r=r,
            q=q
        )
        return sigma

    except Exception:
        return np.nan


# Implied volatility vector for several tickers
def get_implied_vol_vector(tickers, risk_free_rate, target_days=30):
    """
    Computes the ATM implied volatility vector for a list of tickers.

    Note
    ----
    Implied volatilities are read from the *current* live option chain
    (yfinance has no historical option chain endpoint). This vector is
    therefore a point-in-time snapshot as of the retrieval date below,
    not a value consistent with the historical sample period used for
    the correlation matrix it gets combined with.
    """

    print(f"Implied volatilities retrieved as of: {dt.date.today().isoformat()}")
    implied_vols = []
    for ticker in tickers:
        sigma = get_atm_implied_vol(
            ticker=ticker,
            r=risk_free_rate,
            target_days=target_days
        )
        implied_vols.append(sigma)
    return np.array(implied_vols)


# Frozen implied volatility snapshot (load if present, else fetch live and freeze)
def load_or_fetch_implied_vols(tickers, risk_free_rate, target_days, snapshot_path):
    """
    Loads a frozen implied volatility snapshot from `snapshot_path` if it exists,
    otherwise fetches one live snapshot via `get_implied_vol_vector` and saves it
    to `snapshot_path` for every later run to reuse.

    Implied volatilities can only be read from yfinance's *current* live option
    chain (no historical endpoint - see `get_implied_vol_vector`), so without
    freezing, every notebook re-run would silently pull a different market
    snapshot and shift any result downstream of it (Sigma_impl, the lambda grid,
    the implied/hybrid efficient frontiers). Delete `snapshot_path` to force a
    fresh live pull and re-freeze on a new date.
    """

    if snapshot_path.exists():
        snapshot = pd.read_csv(snapshot_path, comment="#", index_col="Ticker")
        with open(snapshot_path) as f:
            header = f.readline()
        print(f"Implied volatilities loaded from frozen snapshot ({header.lstrip('# ').strip()})")
        return snapshot["Implied Volatility"].reindex(tickers)

    implied_vols = get_implied_vol_vector(
        tickers=tickers,
        risk_free_rate=risk_free_rate,
        target_days=target_days
    )
    implied_vols = pd.Series(implied_vols, index=tickers, name="Implied Volatility")

    snapshot_date = dt.date.today().isoformat()
    with open(snapshot_path, "w") as f:
        f.write(f"# Implied volatility snapshot retrieved on {snapshot_date}\n")
        implied_vols.rename_axis("Ticker").reset_index().to_csv(f, index=False)

    print(f"Implied volatilities fetched live and frozen to snapshot: {snapshot_date}")
    return implied_vols