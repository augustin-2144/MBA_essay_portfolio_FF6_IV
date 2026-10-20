import numpy as np
import pandas as pd

from optimization import (
    portfolio_return,
    portfolio_volatility,
    portfolio_sharpe,
    optimize_min_variance,
    optimize_max_sharpe,
    optimize_target_return
)
from array_utils import to_array


# Compute efficient frontier
def compute_efficient_frontier(mu,Sigma,n_points=50,bounds=None):
    mu_ = to_array(mu)

    target_returns = np.linspace(
        np.min(mu_),
        np.max(mu_),
        n_points
    )

    frontier_returns = []
    frontier_volatilities = []
    frontier_weights = []

    for target_return in target_returns:

        weights = optimize_target_return(
            mu=mu,
            Sigma=Sigma,
            target_return=target_return,
            bounds=bounds
        )

        if weights is None:
            continue

        frontier_returns.append(
            portfolio_return(weights, mu)
        )

        frontier_volatilities.append(
            portfolio_volatility(weights, Sigma)
        )

        frontier_weights.append(weights)

    frontier = pd.DataFrame({
        "Return": frontier_returns,
        "Volatility": frontier_volatilities,
        "Weights": frontier_weights
    })

    return frontier


# Frontier summary
def compute_frontier_summary(mu,
                             Sigma,
                             risk_free_rate,
                             n_points=50,
                             bounds=None):
    """
    Computes the efficient frontier together with
    the minimum variance and tangent portfolios.
    """

    frontier = compute_efficient_frontier(
        mu=mu,
        Sigma=Sigma,
        n_points=n_points,
        bounds=bounds
    )

    # Minimum variance portfolio
    w_min_var = optimize_min_variance(
        mu=mu,
        Sigma=Sigma,
        bounds=bounds
    )

    # Tangency portfolio
    w_tangent = optimize_max_sharpe(
        mu=mu,
        Sigma=Sigma,
        risk_free_rate=risk_free_rate,
        bounds=bounds
    )

    # Summary dictionary
    summary = {

        "frontier": frontier,

        "w_min_var": w_min_var,
        "min_var_return": portfolio_return(w_min_var, mu),
        "min_var_volatility": portfolio_volatility(w_min_var, Sigma),
        "min_var_sharpe": portfolio_sharpe(
            w_min_var,
            mu,
            Sigma,
            risk_free_rate
        ),

        "w_tangent": w_tangent,
        "tangent_return": portfolio_return(w_tangent, mu),
        "tangent_volatility": portfolio_volatility(w_tangent, Sigma),
        "tangent_sharpe": portfolio_sharpe(
            w_tangent,
            mu,
            Sigma,
            risk_free_rate
        )
    }

    return summary