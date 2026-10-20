# Stress-testing the hybrid covariance matrix against alternative implied volatility snapshots
import pandas as pd

from covariance import compute_shrunk_implied_covariance
from portfolio_builders import build_optimized_portfolio
from portfolio_metrics import portfolio_summary

# Applies a controlled shock to a vector of implied volatilities
def apply_iv_shock(implied_vols, shock_type, multiplier=None, tickers=None):
    """
    Applies a controlled shock to a vector of implied volatilities, for
    stress-testing the hybrid covariance matrix Sigma_lambda against
    alternative IV snapshots.

    Parameters
    ----------
    implied_vols : pd.Series
        Baseline annualized implied volatilities, indexed by ticker.
    shock_type : str
        One of:
        - "uniform" : multiply every ticker by `multiplier`.
        - "subset"  : multiply only `tickers` by `multiplier`, leaving the
          rest of the universe unchanged.
        - "invert"  : reassign the same set of IV values across tickers in
          reverse rank order (the least volatile asset receives the highest
          value and vice versa), without altering the underlying
          distribution of values.
    multiplier : float, optional
        Multiplicative shock applied for the "uniform" and "subset" shock
        types.
    tickers : list of str, optional
        Subset of tickers shocked for the "subset" shock type.

    Returns
    -------
    pd.Series
        Shocked implied volatilities, indexed identically to `implied_vols`.
    """
    shocked = implied_vols.copy()

    if shock_type == "uniform":
        shocked *= multiplier
    elif shock_type == "subset":
        shocked.loc[tickers] *= multiplier
    elif shock_type == "invert":
        tickers_desc = implied_vols.sort_values(ascending=False).index
        values_asc = implied_vols.sort_values(ascending=True).values
        shocked = pd.Series(values_asc, index=tickers_desc).reindex(implied_vols.index)
    else:
        raise ValueError(f"Unknown shock_type: {shock_type!r}")

    return shocked


# Runs the full lambda-grid optimization under one shocked IV scenario
def run_stress_scenario(
    returns,
    tickers,
    mu,
    implied_vols,
    lambda_grid,
    risk_free_rate,
    bounds,
    shock_type,
    multiplier=None,
    shock_tickers=None
):
    """
    Re-runs the hybrid covariance / max-Sharpe optimization pipeline across
    the full lambda grid, using a shocked implied volatility vector instead
    of the baseline snapshot. Everything else (historical returns, factor
    expected returns, correlation structure) is held fixed, isolating the
    effect of the IV assumption on the resulting allocations and metrics.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns (unchanged across scenarios).
    tickers : list of str
        Investment universe.
    mu : pd.Series
        Expected returns (unchanged across scenarios).
    implied_vols : pd.Series
        Baseline implied volatilities to be shocked.
    lambda_grid : array-like
        Shrinkage grid to sweep, e.g. np.arange(0, 1.05, 0.05).
    risk_free_rate : float
        Annualized risk-free rate.
    bounds : sequence
        Per-asset weight bounds passed to the optimizer.
    shock_type, multiplier, shock_tickers
        Forwarded to `apply_iv_shock`.

    Returns
    -------
    pd.DataFrame
        One row per lambda value (indexed by `Lambda`), same columns as
        `portfolio_summary`.
    """
    iv_shocked = apply_iv_shock(implied_vols, shock_type, multiplier, shock_tickers)

    results = []
    for lambda_ in lambda_grid:
        w_lambda, Sigma_lambda = _optimize_for_shocked_iv(
            returns, tickers, mu, iv_shocked, lambda_, risk_free_rate, bounds
        )

        summary_lambda = portfolio_summary(
            tickers=tickers,
            weights=w_lambda.values,
            mu=mu,
            Sigma=Sigma_lambda,
            risk_free_rate=risk_free_rate
        )
        summary_lambda["Lambda"] = lambda_
        results.append(summary_lambda)

    return pd.DataFrame(results).set_index("Lambda")


# Shared optimization step for one (already-shocked) IV vector at one lambda
def _optimize_for_shocked_iv(returns, tickers, mu, iv_shocked, lambda_, risk_free_rate, bounds):
    Sigma_lambda = compute_shrunk_implied_covariance(
        returns=returns,
        implied_vols=iv_shocked,
        lambda_=lambda_
    )
    w_lambda = build_optimized_portfolio(
        tickers=tickers,
        mu=mu,
        Sigma=Sigma_lambda,
        risk_free_rate=risk_free_rate,
        name=f"Lambda {lambda_:.2f}",
        bounds=bounds
    )
    return w_lambda, Sigma_lambda


# Optimized weights for one shocked scenario at a single lambda value
def get_stress_scenario_weights(
    returns, tickers, mu, implied_vols, lambda_, risk_free_rate, bounds,
    shock_type, multiplier=None, shock_tickers=None
):
    """
    Computes the optimized portfolio weights for one shocked IV scenario at a
    single lambda value - a lighter-weight counterpart to `run_stress_scenario`
    for inspecting the allocation mechanism behind a specific scenario/lambda
    combination, without sweeping the full grid.

    Returns
    -------
    pd.Series
        Optimized weights, indexed by ticker.
    """
    iv_shocked = apply_iv_shock(implied_vols, shock_type, multiplier, shock_tickers)
    w_lambda, _ = _optimize_for_shocked_iv(
        returns, tickers, mu, iv_shocked, lambda_, risk_free_rate, bounds
    )
    return w_lambda
