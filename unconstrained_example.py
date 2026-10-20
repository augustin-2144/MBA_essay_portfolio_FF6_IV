"""
Closed-form (unconstrained) tangency portfolio, used in Chapter 2 to illustrate the
sensitivity of the explicit mean-variance solution to estimation error, before the
long-only constraint is introduced.
"""
import numpy as np
import pandas as pd

from array_utils import to_array


def closed_form_tangency_weights(mu, Sigma, risk_free_rate):
    """
    Unconstrained tangency portfolio weights, in closed form :

        w* = Sigma^-1 (mu - rf * 1) / (1' Sigma^-1 (mu - rf * 1))

    No long-only / bounds constraint is applied here : short positions and leverage
    are both allowed, unlike `optimize_max_sharpe` used everywhere else in this study.
    """

    mu_ = to_array(mu)
    Sigma_ = to_array(Sigma)

    ones = np.ones(len(mu_))
    excess = mu_ - risk_free_rate * ones

    Sigma_inv = np.linalg.inv(Sigma_)
    w = (Sigma_inv @ excess) / (ones @ Sigma_inv @ excess)

    if hasattr(mu, "index"):
        return pd.Series(w, index=mu.index)
    return w