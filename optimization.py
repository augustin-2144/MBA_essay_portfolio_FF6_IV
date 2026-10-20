import numpy as np
from scipy.optimize import minimize

from array_utils import to_array

# Portfolio return
def portfolio_return(weights, mu):
    return float(weights @ to_array(mu))


# Portfolio volatility
def portfolio_volatility(weights, Sigma):
    Sigma_ = to_array(Sigma)
    return float(np.sqrt(weights @ Sigma_ @ weights))


# Portfolio Sharpe ratio
def portfolio_sharpe(weights, mu, Sigma, risk_free_rate):
    vol = portfolio_volatility(weights, Sigma)
    if vol == 0:
        return np.nan
    ret = portfolio_return(weights, mu)
    return (ret - risk_free_rate) / vol


# Negative Sharpe ratio
def negative_sharpe(weights, mu, Sigma, risk_free_rate):
    """
    Negative Sharpe ratio for optimization.
    """

    return -portfolio_sharpe(
        weights,
        mu,
        Sigma,
        risk_free_rate
    )


# Shared SLSQP setup for the three optimizers below: budget (+ optional ESG, + optional extra)
# constraints, default long-only bounds, equal-weight initial guess. Returns the raw OptimizeResult
# so each caller can decide how to handle failure (they don't all agree: see optimize_target_return).
def _run_mean_variance_optimization(objective, objective_args, mu, bounds=None,
                                     esg_scores=None, min_esg_score=None, extra_constraints=None):
    n_assets = len(mu)

    if bounds is None:
        # guarantee that weights are between 0 and 1, and respect the long-only constraint
        bounds = [(0, 1)] * n_assets

    constraints = [{"type": "eq", "fun": lambda w: np.sum(w) - 1}]

    if extra_constraints:
        constraints.extend(extra_constraints)

    if esg_scores is not None and min_esg_score is not None:
        constraints.append(
            make_min_esg_constraint(esg_scores, min_esg_score)
        )

    init = np.ones(n_assets) / n_assets

    return minimize(
        objective,
        init,
        args=objective_args,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )


# Minimum variance optimization
def optimize_min_variance(mu, Sigma, bounds=None, esg_scores=None, min_esg_score=None):
    result = _run_mean_variance_optimization(
        portfolio_volatility, (Sigma,), mu, bounds, esg_scores, min_esg_score
    )

    if not result.success:
        raise ValueError(
            f"Minimum variance optimization failed: {result.message}"
        )

    return result.x


# Maximum Sharpe optimization
def optimize_max_sharpe(mu, Sigma, risk_free_rate, bounds=None, esg_scores=None, min_esg_score=None):
    result = _run_mean_variance_optimization(
        negative_sharpe, (mu, Sigma, risk_free_rate), mu, bounds, esg_scores, min_esg_score
    )

    if not result.success:
        raise ValueError(
            f"Maximum Sharpe optimization failed: {result.message}"
        )

    return result.x


# Target return optimization
def optimize_target_return(mu, Sigma, target_return, bounds=None, esg_scores=None, min_esg_score=None):
    return_constraint = {"type": "eq", "fun": lambda w: portfolio_return(w, mu) - target_return}

    result = _run_mean_variance_optimization(
        portfolio_volatility, (Sigma,), mu, bounds, esg_scores, min_esg_score,
        extra_constraints=[return_constraint]
    )

    if not result.success:
        return None

    return result.x

# ESG minimum score constraint
def make_min_esg_constraint(esg_scores, min_esg_score):
    """
    Builds a scipy-compatible ESG constraint.

    Constraint:
        w'ESG >= min_esg_score
    """

    scores = to_array(esg_scores)
    return {
        "type": "ineq",
        "fun": lambda w: float(w @ scores - min_esg_score)
    }