import pandas as pd
from optimization import optimize_max_sharpe


# Build one optimized portfolio
def build_optimized_portfolio(
    tickers,
    mu,
    Sigma,
    risk_free_rate,
    name="Portfolio",
    bounds=None,
    esg_scores=None,
    min_esg_score=None
):
    weights = optimize_max_sharpe(
        mu=mu,
        Sigma=Sigma,
        risk_free_rate=risk_free_rate,
        bounds=bounds,
        esg_scores=esg_scores,
        min_esg_score=min_esg_score
    )

    weights = pd.Series(
        weights,
        index=tickers,
        name=name
    )

    return weights