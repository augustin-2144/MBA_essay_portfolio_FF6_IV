# Handles covariance matrix construction (historical and implied)
import numpy as np
import pandas as pd

from config import FREQUENCY

# Historical covariance
def compute_historical_covariance(returns, frequency=FREQUENCY):
    return returns.cov() * frequency

# Implied covariance
def compute_implied_covariance(returns, implied_vols):
    corr = returns.corr()
    implied_vols = pd.Series(
        implied_vols,
        index=returns.columns
    )
    D = np.diag(implied_vols.values)
    Sigma = D @ corr.values @ D
    return pd.DataFrame(
        Sigma,
        index=returns.columns,
        columns=returns.columns
    )

# Shrunk covariance (convex combination of historical and implied)
def compute_shrunk_implied_covariance(returns, implied_vols, lambda_):
    """
    Compute a shrinkage covariance matrix between historical covariance
    and implied-volatility-based covariance.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset returns.
    implied_vols : array-like or pd.Series
        Implied volatilities for each asset, annualized.
    lambda_ : float
        Shrinkage intensity toward the implied covariance matrix.
        lambda_=0 gives the historical covariance.
        lambda_=1 gives the implied-volatility-based covariance.

    Returns
    -------
    pd.DataFrame
        Shrunk covariance matrix.
    """

    if not 0 <= lambda_ <= 1:
        raise ValueError("lambda_ must be between 0 and 1.")

    Sigma_hist = compute_historical_covariance(returns)
    Sigma_impl = compute_implied_covariance(returns, implied_vols)

    Sigma_shrunk = (1 - lambda_) * Sigma_hist + lambda_ * Sigma_impl

    return Sigma_shrunk