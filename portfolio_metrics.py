import numpy as np
import pandas as pd

from config import FREQUENCY
from array_utils import to_array

# Basic portfolio performance
def portfolio_metrics(weights, mu, Sigma, risk_free_rate):
    mu_ = to_array(mu)
    Sigma_ = to_array(Sigma)
    portfolio_return = weights @ mu_
    portfolio_volatility = np.sqrt(weights @ Sigma_ @ weights)
    portfolio_sharpe = (portfolio_return - risk_free_rate) / portfolio_volatility
    return portfolio_return, portfolio_volatility, portfolio_sharpe


# Raw Herfindahl-Hirschman Index
def hhi(weights):
    weights = np.asarray(weights)
    return np.sum(weights**2)


# Normalized Herfindahl-Hirschman Index
def hhi_normalized(weights):
    weights = np.asarray(weights)
    n = len(weights)
    hhi_raw = hhi(weights)
    return (hhi_raw - 1/n) / (1 - 1/n)


# Effective number of positions
def effective_number_of_positions(weights):
    return 1 / hhi(weights)


# Portfolio exposure
def portfolio_exposure(weights):
    weights = np.asarray(weights)
    long_exposure = weights[weights > 0].sum()
    short_exposure = weights[weights < 0].sum()
    gross_exposure = np.abs(weights).sum()
    net_exposure = weights.sum()
    exposure = pd.Series({
        "Long Exposure": long_exposure,
        "Short Exposure": short_exposure,
        "Gross Exposure": gross_exposure,
        "Net Exposure": net_exposure
    })
    return exposure


# Risk contributions
def risk_contributions(weights, Sigma):
    """
    Computes absolute risk contributions to portfolio volatility.
    """

    Sigma_ = to_array(Sigma)
    weights = np.asarray(weights)
    marginal_risk = Sigma_ @ weights
    portfolio_volatility = np.sqrt(weights @ Sigma_ @ weights)
    if portfolio_volatility == 0:
        return np.full_like(weights, np.nan)
    contributions = weights * marginal_risk / portfolio_volatility
    return contributions


# Risk contribution table
def risk_contribution_table(tickers, weights, Sigma):
    rc = risk_contributions(weights, Sigma)
    table = pd.DataFrame(
        {"Risk Contribution": rc,
        "Risk Contribution (%)": rc / np.nansum(rc)},
        index=tickers
    )
    table.index.name = "Asset"
    return table


# Annualized performance from a realized return series (ex-post)
def annualized_stats(returns, risk_free_rate, frequency=FREQUENCY):
    """
    Computes annualized return, volatility and Sharpe ratio from a realized
    (ex-post) return series, e.g. a backtest's realized portfolio returns.

    `risk_free_rate` can be a scalar (annualized rate, subtracted from the
    annualized portfolio return - as before), or a pandas Series of period
    risk-free rates aligned to `returns.index`. In the latter case the
    Sharpe ratio is computed from the period-by-period excess return series
    (returns - risk_free_rate), annualized directly, rather than by
    subtracting two separately annualized aggregates - the more precise
    convention when the risk-free rate moves materially within the window
    being evaluated (e.g. a multi-year out-of-sample backtest).
    """
    ann_return = (1 + returns.mean()) ** frequency - 1
    ann_vol = returns.std() * np.sqrt(frequency)

    if np.ndim(risk_free_rate) > 0:
        excess = returns - risk_free_rate.reindex(returns.index)
        ann_excess_return = (1 + excess.mean()) ** frequency - 1
        excess_vol = excess.std() * np.sqrt(frequency)
        sharpe = ann_excess_return / excess_vol
    else:
        sharpe = (ann_return - risk_free_rate) / ann_vol

    return ann_return, ann_vol, sharpe


# Rolling annualized Sharpe ratio from a realized return series
def rolling_sharpe_ratio(returns, risk_free_rate, window=12, frequency=FREQUENCY):
    """
    Computes a rolling annualized Sharpe ratio over a fixed-length window of
    realized returns. The first `window - 1` entries are NaN.

    `risk_free_rate` can be a scalar or a pandas Series of period risk-free
    rates aligned to `returns.index` (see `annualized_stats`); in the latter
    case each window's Sharpe ratio is computed from that window's own
    period-by-period excess returns, rather than against a single rate
    averaged over the whole series.
    """
    if np.ndim(risk_free_rate) > 0:
        excess = returns - risk_free_rate.reindex(returns.index)
        rolling_mean = excess.rolling(window).mean()
        rolling_std = excess.rolling(window).std()
        ann_excess_return = (1 + rolling_mean) ** frequency - 1
        ann_vol = rolling_std * np.sqrt(frequency)
        return ann_excess_return / ann_vol
    else:
        rolling_mean = returns.rolling(window).mean()
        rolling_std = returns.rolling(window).std()
        ann_return = (1 + rolling_mean) ** frequency - 1
        ann_vol = rolling_std * np.sqrt(frequency)
        return (ann_return - risk_free_rate) / ann_vol


# Sortino ratio from a realized return series
def sortino_ratio(returns, risk_free_rate, frequency=FREQUENCY):
    """
    Computes the annualized Sortino ratio from a realized return series,
    using the risk-free rate as the minimum acceptable return (MAR) for the
    downside deviation - consistent with how the Sharpe ratio above treats
    the risk-free rate as the return threshold.

    `risk_free_rate` can be a scalar or a pandas Series of period rates
    aligned to `returns.index` (see `annualized_stats`); in the latter case
    the MAR varies period by period instead of being held fixed at a single
    averaged rate.
    """
    if np.ndim(risk_free_rate) > 0:
        rf_period = risk_free_rate.reindex(returns.index)
        excess = returns - rf_period
        ann_excess_return = (1 + excess.mean()) ** frequency - 1

        downside = excess[excess < 0]
        downside_deviation = np.sqrt((downside ** 2).mean()) * np.sqrt(frequency)

        if downside_deviation == 0 or pd.isna(downside_deviation):
            return np.nan

        return ann_excess_return / downside_deviation
    else:
        ann_return = (1 + returns.mean()) ** frequency - 1
        rf_monthly = (1 + risk_free_rate) ** (1 / frequency) - 1

        downside = returns[returns < rf_monthly] - rf_monthly
        downside_deviation = np.sqrt((downside ** 2).mean()) * np.sqrt(frequency)

        if downside_deviation == 0:
            return np.nan

        return (ann_return - risk_free_rate) / downside_deviation


# Cumulative return from a realized return series
def cumulative_return(returns):
    """
    Computes the total (non-annualized) cumulative return compounded over
    the full realized return series.
    """
    return (1 + returns).prod() - 1


# Drawdown series from a realized return series (underwater curve)
def drawdown_series(returns):
    """
    Computes the full drawdown series of the cumulative wealth curve implied
    by a realized return series: at each date, the return of wealth(t)
    relative to its running maximum so far.
    """
    wealth = (1 + returns).cumprod()
    return wealth / wealth.cummax() - 1


# Maximum drawdown from a realized return series
def max_drawdown(returns):
    """
    Computes the maximum drawdown of the cumulative wealth curve implied by
    a realized return series.
    """
    return drawdown_series(returns).min()


# Full portfolio summary
def portfolio_summary(tickers, weights, mu, Sigma, risk_free_rate):
    portfolio_return, portfolio_volatility, portfolio_sharpe = portfolio_metrics(
        weights=weights,
        mu=mu,
        Sigma=Sigma,
        risk_free_rate=risk_free_rate
    )
    summary = pd.Series({
        "Expected Return": round(portfolio_return*100,2),
        "Volatility": round(portfolio_volatility*100,2),
        "Sharpe Ratio": portfolio_sharpe,
        "HHI Normalized": hhi_normalized(weights),
        "Effective Number of Positions": effective_number_of_positions(weights),
        "Long Exposure": round(portfolio_exposure(weights)["Long Exposure"]*100,2),
        "Short Exposure": round(portfolio_exposure(weights)["Short Exposure"]*100,2),
        "Gross Exposure": round(portfolio_exposure(weights)["Gross Exposure"]*100,2),
        "Net Exposure": round(portfolio_exposure(weights)["Net Exposure"]*100,2)
    })
    return summary


# Distance between two weight vectors: amount and concentration of the shift
def weight_shift_metrics(w_from, w_to):
    """
    Measures how much, and how concentratedly, a portfolio's weights move
    from one allocation to another. Two uses in this study: turnover from
    one rebalancing date to the next (w_from/w_to = consecutive months),
    and sensitivity of a stress-tested scenario to its baseline at a fixed
    lambda (w_from/w_to = baseline/scenario weights).

    Parameters
    ----------
    w_from, w_to : pd.Series
        Two weight vectors over the same set of assets, sharing the same
        index.

    Returns
    -------
    pd.Series with:
        "Turnover" : one-way L1 distance between the two vectors
            (half the sum of absolute weight changes) - the fraction of the
            portfolio that would need to be traded to move from `w_from` to
            `w_to`.
        "Effective Number of Positions (Shift)" : HHI-style effective
            number of assets absorbing the reallocation, computed on the
            normalized absolute weight changes. Ranges from 1 (the shift is
            entirely concentrated on a single asset) to n (the shift is
            spread evenly across all n assets that moved).
    """
    delta = (w_to - w_from).abs()
    total_shift = delta.sum()

    turnover = total_shift / 2

    if total_shift == 0:
        enp_shift = np.nan
    else:
        p = delta / total_shift
        enp_shift = 1 / (p ** 2).sum()

    return pd.Series({
        "Turnover": turnover,
        "Effective Number of Positions (Shift)": enp_shift
    })