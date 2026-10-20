# Format portfolio metrics
def format_portfolio_metrics(metrics_table):
    """
    Formats key portfolio metrics for notebook display.
    """

    formatted = metrics_table.copy()

    percent_cols = [
        "Expected Return",
        "Volatility",
        "Long Exposure",
        "Short Exposure",
        "Gross Exposure",
        "Net Exposure",
        "Max Drawdown",
        "Cumulative Return"
    ]

    for col in percent_cols:
        if col in formatted.columns:
            formatted[col] = formatted[col].round(2)

    ratio_cols = ["Sharpe Ratio", "Sortino Ratio", "Skewness", "Kurtosis"]

    for col in ratio_cols:
        if col in formatted.columns:
            formatted[col] = formatted[col].round(3)

    if "HHI Normalized" in formatted.columns:
        formatted["HHI Normalized"] = formatted["HHI Normalized"].round(3)

    if "Effective Number of Positions" in formatted.columns:
        formatted["Effective Number of Positions"] = formatted["Effective Number of Positions"].round(2)

    return formatted