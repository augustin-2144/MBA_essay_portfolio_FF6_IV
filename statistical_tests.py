# Ledoit & Wolf (2008) test for the equality of two Sharpe ratios
import numpy as np
import pandas as pd
from scipy.stats import norm

# Sharpe ratio (linear annualization, as required by the Ledoit-Wolf test)
def sharpe_ratio(excess_returns, frequency=12):
    """
    Annualized Sharpe ratio using the standard linear scaling
    mean(excess) / std(excess) * sqrt(frequency).

    This is the exact functional form for which Ledoit & Wolf (2008) prove
    asymptotic normality of the studentized Sharpe-ratio difference. Kept
    here (unused by `ledoit_wolf_test` below) purely as the reference
    definition against which `sharpe_ratio_compounded` is compared - see
    the diagnostic in the notebook and the delta-method derivation in
    Annexe \\ref{bootstrap} of the write-up.
    """
    return np.mean(excess_returns) / np.std(excess_returns) * np.sqrt(frequency)


# Sharpe ratio (compounded annualization, matching `annualized_stats`)
def sharpe_ratio_compounded(returns, risk_free_rate, frequency=12):
    """
    Annualized Sharpe ratio using the same compounded-return convention as
    `annualized_stats` in `portfolio_metrics.py` - (1+mean)^frequency - 1
    for the return, std*sqrt(frequency) for the volatility - so that the
    Ledoit & Wolf test operates on exactly the same Sharpe ratio definition
    reported everywhere else in this study.

    `risk_free_rate` can be a scalar (annualized rate, subtracted from the
    annualized return - as before), or an array-like of period risk-free
    rates the same length as `returns`, positionally aligned. In the latter
    case the Sharpe ratio is computed from the period-by-period excess
    return series (returns - risk_free_rate), matching the Series case of
    `annualized_stats`.

    Asymptotic validity for this compounded variant is established in
    Annexe \\ref{bootstrap} of the write-up via the delta method: the map
    x -> (1+x)^frequency - 1 is a polynomial (C^infty) with a derivative
    that never vanishes for x > -1, so the same argument Ledoit & Wolf use
    for the linear Sharpe ratio carries over unchanged.
    """
    returns = np.asarray(returns)

    if np.ndim(risk_free_rate) > 0:
        excess = returns - np.asarray(risk_free_rate)
        ann_return = (1 + np.mean(excess)) ** frequency - 1
        ann_vol = np.std(excess, ddof=1) * np.sqrt(frequency)
        return ann_return / ann_vol
    else:
        ann_return = (1 + np.mean(returns)) ** frequency - 1
        ann_vol = np.std(returns, ddof=1) * np.sqrt(frequency)
        return (ann_return - risk_free_rate) / ann_vol


# Politis & Romano (1994) stationary bootstrap - one resampled path of indices
def stationary_bootstrap_indices(T, b, rng):
    """
    Draws one stationary-bootstrap resampling path of length T (indices into
    a series of length T), with mean block length b.

    At each step, with probability p = 1 - 1/b the block continues (next
    index in sequence, wrapping around circularly); otherwise a new random
    starting index is drawn. Block lengths are therefore geometric with
    mean b, as in Politis & Romano (1994).
    """
    p = 1 - 1/b
    indices = np.empty(T, dtype=int)
    indices[0] = rng.integers(T)
    for t in range(1, T):
        if rng.random() < p:
            indices[t] = (indices[t - 1] + 1) % T
        else:
            indices[t] = rng.integers(T)
    return indices


# Ledoit & Wolf (2008) Sharpe ratio comparison test
def ledoit_wolf_test(r_bench, r_ff6, rf, n_bootstrap=1000, frequency=12, seed=42, ratio_fn=None):
    """
    Tests H0: ratio(r_bench) = ratio(r_ff6) using the studentized stationary
    bootstrap of Ledoit & Wolf (2008), with block resampling a la
    Politis & Romano (1994). Both series are resampled along the SAME
    bootstrap path at each replication, to preserve their cross-correlation.

    `ratio_fn` selects which risk-adjusted ratio is compared; it defaults to
    `sharpe_ratio_compounded` (same compounded-annualization convention as
    `annualized_stats`, used everywhere else in this study) rather than the
    linear Sharpe ratio of Ledoit & Wolf's original paper - see
    `sharpe_ratio_compounded`'s docstring and Annexe \\ref{bootstrap} of the
    write-up for the delta-method argument justifying this substitution.
    Passing `sortino_ratio` instead reuses the exact same bootstrap
    machinery to test the Sortino-ratio gap - both functions share the same
    `(returns, risk_free_rate, frequency)` signature; `returns`/`risk_free_rate`
    are wrapped as same-length pandas Series with a shared default index
    before every call, so `sortino_ratio`'s array-mode `.reindex()` resolves
    to a trivial positional match, keeping the substitution a drop-in one.
    The delta-method argument in Annexe \\ref{bootstrap} does
    not itself cover the Sortino ratio (its downside-deviation denominator
    is not a smooth function of the return moments in the same way), so
    this variant should be read as an informal robustness check rather than
    a result with the same asymptotic guarantee as the Sharpe-based test.

    `rf` is passed through period-by-period (not pre-averaged into a single
    scalar): both the observed statistic and every bootstrap resample net
    each return against its own contemporaneous risk-free rate, consistent
    with the main backtest table and rolling Sharpe. `rf` is resampled along
    the SAME bootstrap path as `r_bench`/`r_ff6` at each replication, so
    that a resampled return always stays paired with its own period's rate.

    Parameters
    ----------
    r_bench, r_ff6 : array-like
        Monthly return series of equal length, observation-aligned.
    rf : array-like
        Monthly risk-free rate, same length and dates.
    n_bootstrap : int
        Number of bootstrap resamples (B).
    frequency : int
        Periods per year used to annualize the ratio.
    seed : int
        Random seed for reproducibility.
    ratio_fn : callable, optional
        Ratio function with signature `(returns, risk_free_rate, frequency)`.
        Defaults to `sharpe_ratio_compounded`.

    Returns
    -------
    dict with SR_bench, SR_ff6, theta_hat, t_stat, p_value, and the raw
    bootstrap draws (theta_boot, SR_bench_boot, SR_ff6_boot) for further
    inspection (e.g. plotting the bootstrap distribution). Keys are named
    "SR_*" regardless of `ratio_fn` for a uniform interface across callers.
    """
    if ratio_fn is None:
        ratio_fn = sharpe_ratio_compounded

    r_bench = np.asarray(r_bench)
    r_ff6 = np.asarray(r_ff6)
    rf = np.asarray(rf)

    T = len(r_bench)
    b = int(T ** (1 / 3))

    sr_bench = ratio_fn(pd.Series(r_bench), pd.Series(rf), frequency=frequency)
    sr_ff6 = ratio_fn(pd.Series(r_ff6), pd.Series(rf), frequency=frequency)
    theta_hat = sr_bench - sr_ff6

    rng = np.random.default_rng(seed)
    theta_boot = np.empty(n_bootstrap)
    sr_bench_boot = np.empty(n_bootstrap)
    sr_ff6_boot = np.empty(n_bootstrap)

    for i in range(n_bootstrap):
        idx = stationary_bootstrap_indices(T, b, rng)
        rf_resampled = pd.Series(rf[idx])
        sr_bench_boot[i] = ratio_fn(pd.Series(r_bench[idx]), rf_resampled, frequency=frequency)
        sr_ff6_boot[i] = ratio_fn(pd.Series(r_ff6[idx]), rf_resampled, frequency=frequency)
        theta_boot[i] = sr_bench_boot[i] - sr_ff6_boot[i]

    var_hat = np.var(theta_boot)
    t_stat = theta_hat / np.sqrt(var_hat)
    p_value = 2 * (1 - norm.cdf(np.abs(t_stat)))

    return {
        "SR_bench": sr_bench,
        "SR_ff6": sr_ff6,
        "theta_hat": theta_hat,
        "t_stat": t_stat,
        "p_value": p_value,
        "theta_boot": theta_boot,
        "SR_bench_boot": sr_bench_boot,
        "SR_ff6_boot": sr_ff6_boot
    }