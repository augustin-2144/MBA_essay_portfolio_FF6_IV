# Rolling out-of-sample backtest: historical vs FF6 factor-based expected returns
import pandas as pd

from config import FREQUENCY
from covariance import compute_historical_covariance
from Fama_French import (
    FF6_FACTORS,
    compute_naive_expected_returns,
    build_ff6_regression_table,
    compute_factor_expected_returns
)
from optimization import optimize_max_sharpe

# Shared per-window estimation for the two rolling backtests below: historical covariance,
# FF6 regression + factor-based expected returns, and the window-average risk-free rate -
# both re-estimated fresh at every rebalancing date from the same trailing window.
def _compute_window_inputs(window, tickers, factors, frequency):
    Sigma_window = compute_historical_covariance(window[tickers], frequency=frequency)

    regression_table_window = build_ff6_regression_table(
        data=window,
        tickers=tickers,
        factors=factors
    )
    mu_factor_window = compute_factor_expected_returns(
        data=window,
        regression_table=regression_table_window,
        factors=factors,
        annualize=True
    )

    rf_window = (1 + window["RF"].mean()) ** frequency - 1

    return Sigma_window, mu_factor_window, rf_window


# Rolling backtest
def rolling_backtest(data, tickers, factors=FF6_FACTORS, lookback=36, bounds=None, frequency=FREQUENCY):
    """
    Rolling out-of-sample backtest comparing two expected-return models that
    share the same historical covariance, both re-estimated on a rolling
    window and rebalanced monthly via max-Sharpe optimization:

        Historical Benchmark : mu = historical mean,      Sigma = historical covariance
        Factor-Based (FF6)   : mu = FF6 factor-based,      Sigma = historical covariance

    The risk-free rate used inside the optimizer is the average of `data["RF"]`
    over the same rolling window so each rebalancing only uses information
    available up to that date.

    Parameters
    ----------
    data : pd.DataFrame
        Asset returns (columns = tickers) aligned with FF6 factors and "RF",
        as produced by `align_data`.
    tickers : list of str
    factors : list of str
    lookback : int
        Estimation window length, in months.
    bounds : list of (float, float), optional
        Per-asset weight bounds forwarded to `optimize_max_sharpe`.

    Returns
    -------
    returns : pd.DataFrame
        Realized next-month return for each configuration, indexed by
        rebalancing date.
    weights : dict[str, pd.DataFrame]
        Realized portfolio weights for each configuration, indexed by
        rebalancing date.
    """

    dates = []
    returns_hist, returns_factor = [], []
    weights_hist, weights_factor = {}, {}

    for t in range(lookback, len(data)):
        window = data.iloc[t - lookback:t]
        realized_next = data[tickers].iloc[t]
        rebalance_date = data.index[t]

        Sigma_window, mu_factor_window, rf_window = _compute_window_inputs(
            window, tickers, factors, frequency
        )

        mu_naive_window = compute_naive_expected_returns(
            data=window,
            tickers=tickers,
            annualize=True
        )

        w_hist = optimize_max_sharpe(
            mu=mu_naive_window,
            Sigma=Sigma_window,
            risk_free_rate=rf_window,
            bounds=bounds
        )
        w_factor = optimize_max_sharpe(
            mu=mu_factor_window,
            Sigma=Sigma_window,
            risk_free_rate=rf_window,
            bounds=bounds
        )

        returns_hist.append(w_hist @ realized_next.values)
        returns_factor.append(w_factor @ realized_next.values)
        dates.append(rebalance_date)

        weights_hist[rebalance_date] = pd.Series(w_hist, index=tickers)
        weights_factor[rebalance_date] = pd.Series(w_factor, index=tickers)

    returns = pd.DataFrame({
        "Historical Benchmark": returns_hist,
        "Factor-Based (FF6)": returns_factor
    }, index=pd.Index(dates, name="Date"))

    weights = {
        "Historical Benchmark": pd.DataFrame(weights_hist).T,
        "Factor-Based (FF6)": pd.DataFrame(weights_factor).T
    }

    return returns, weights


# Rolling backtest with an ESG minimum-score constraint, at several thresholds
def rolling_backtest_esg(data, tickers, esg_scores, min_esg_scores, factors=FF6_FACTORS, lookback=36, bounds=None, frequency=FREQUENCY):
    """
    Rolling out-of-sample backtest isolating the effect of the ESG minimum-score
    constraint (`make_min_esg_constraint`) on the Factor-Based (FF6) configuration
    of `rolling_backtest` - same expected-return model (FF6), same historical
    covariance, same 36-month rolling window and monthly max-Sharpe rebalancing,
    with the ESG constraint switched on at each threshold in `min_esg_scores`
    (unconstrained baseline included as its own column).

    All configurations share the same estimation window at each rebalancing date
    (mu, Sigma and the risk-free rate are computed once per date and reused
    across the baseline and every threshold), so any difference between columns
    is attributable to the ESG constraint alone.

    Parameters
    ----------
    data : pd.DataFrame
        Asset returns aligned with FF6 factors and "RF", as produced by `align_data`.
    tickers : list of str
    esg_scores : pd.Series
        ESG score per ticker (LSEG 0-5 scale), aligned with `tickers`.
    min_esg_scores : list of float
        ESG thresholds to test, each producing its own column.
    factors : list of str
    lookback : int
        Estimation window length, in months.
    bounds : list of (float, float), optional
        Per-asset weight bounds forwarded to `optimize_max_sharpe`.

    Returns
    -------
    returns : pd.DataFrame
        Realized next-month return for the unconstrained baseline and each ESG
        threshold, indexed by rebalancing date.
    weights : dict[str, pd.DataFrame]
        Realized portfolio weights for each configuration, indexed by
        rebalancing date.
    """

    labels = ["Factor-Based (FF6)"] + [f"ESG ≥ {thr}" for thr in min_esg_scores]

    dates = []
    returns_by_label = {label: [] for label in labels}
    weights_by_label = {label: {} for label in labels}

    for t in range(lookback, len(data)):
        window = data.iloc[t - lookback:t]
        realized_next = data[tickers].iloc[t]
        rebalance_date = data.index[t]

        Sigma_window, mu_factor_window, rf_window = _compute_window_inputs(
            window, tickers, factors, frequency
        )

        w_baseline = optimize_max_sharpe(
            mu=mu_factor_window,
            Sigma=Sigma_window,
            risk_free_rate=rf_window,
            bounds=bounds
        )
        returns_by_label["Factor-Based (FF6)"].append(w_baseline @ realized_next.values)
        weights_by_label["Factor-Based (FF6)"][rebalance_date] = pd.Series(w_baseline, index=tickers)

        for thr, label in zip(min_esg_scores, labels[1:]):
            w_esg = optimize_max_sharpe(
                mu=mu_factor_window,
                Sigma=Sigma_window,
                risk_free_rate=rf_window,
                bounds=bounds,
                esg_scores=esg_scores,
                min_esg_score=thr
            )
            returns_by_label[label].append(w_esg @ realized_next.values)
            weights_by_label[label][rebalance_date] = pd.Series(w_esg, index=tickers)

        dates.append(rebalance_date)

    returns = pd.DataFrame(returns_by_label, index=pd.Index(dates, name="Date"))
    weights = {label: pd.DataFrame(weights_by_label[label]).T for label in labels}

    return returns, weights