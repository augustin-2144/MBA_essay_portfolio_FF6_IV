# Fama-French 6-factor data loading, regressions, and factor-based expected returns
import datetime as dt
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pandas_datareader.famafrench import FamaFrenchReader

from config import FREQUENCY, START_DATE, END_DATE, FF6_FACTORS_SNAPSHOT_PATH
from visualization import set_plot_style, hide_spines

FF6_FACTORS = ["Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"]

# Download Fama-French 6-factor data
def download_ff6_factors(start, end):
    """
    Downloads Fama-French 5 factors and Momentum factor, then merges them into a single 6-factor dataframe.

    Returns
    -------
    pd.DataFrame
        Monthly Fama-French 6-factor data, expressed in decimal returns.
    """

    ff5 = FamaFrenchReader(
        "F-F_Research_Data_5_Factors_2x3",
        start=start,
        end=end
    ).read()[0]

    ff5 = ff5 / 100
    ff5.index = ff5.index.to_timestamp(how="end").to_period("M")
    ff5.columns = ff5.columns.str.strip()

    momentum = FamaFrenchReader(
        "F-F_Momentum_Factor",
        start=start,
        end=end
    ).read()[0]

    momentum = momentum / 100
    momentum.index = momentum.index.to_timestamp(how="end").to_period("M")
    momentum.columns = momentum.columns.str.strip()

    ff6 = ff5.join(momentum[["Mom"]], how="inner")
    ff6.columns = ff6.columns.str.strip()
    return ff6

# Frozen Fama-French factor snapshot (load if present, else fetch live and freeze)
def load_or_fetch_ff6_factors(start, end, snapshot_path):
    """
    Loads a frozen Fama-French 6-factor snapshot from `snapshot_path` if it
    exists, otherwise downloads factors live via `download_ff6_factors` and
    saves them to `snapshot_path` for every later run to reuse.

    Ken French's data library is periodically restated even for months already
    published, so without freezing, re-running the notebook weeks apart can
    silently shift mu_factor and every result built on it. Delete
    `snapshot_path` to force a fresh live pull and re-freeze on a new date.
    """

    if snapshot_path.exists():
        ff6 = pd.read_csv(snapshot_path, comment="#", index_col=0)
        ff6.index = pd.PeriodIndex(ff6.index, freq="M")
        with open(snapshot_path) as f:
            header = f.readline()
        print(f"Fama-French factors loaded from frozen snapshot ({header.lstrip('# ').strip()})")
        return ff6

    ff6 = download_ff6_factors(start=start, end=end)

    snapshot_date = dt.date.today().isoformat()
    with open(snapshot_path, "w") as f:
        f.write(f"# Fama-French 6-factor snapshot retrieved on {snapshot_date} (start={start}, end={end})\n")
        ff6.to_csv(f)

    print(f"Fama-French factors fetched live and frozen to snapshot: {snapshot_date}")
    return ff6


# Run one Fama-French 6-factor regression
def run_ff6_regression(data, ticker, factors=FF6_FACTORS):
    """
    Runs a Fama-French 6-factor regression for one asset.
    Regression:
        R_i - RF = alpha + beta_1(Mkt-RF) + beta_2(SMB) + ... + beta_6(Mom) + epsilon
    """

    y = data[ticker] - data["RF"]
    X = sm.add_constant(data[factors])
    result = sm.OLS(y, X).fit()
    return result


# Build regression table for several assets
def build_ff6_regression_table(data, tickers, factors=FF6_FACTORS):
    """
    Runs Fama-French 6-factor regressions for several assets and stores
    coefficients, adjusted R², and p-values in a clean dataframe.
    """

    rows = []

    for ticker in tickers:
        result = run_ff6_regression(data, ticker, factors=factors)

        row = {
            "Asset": ticker,
            "alpha": result.params["const"] * 100,  # alpha in % per month
            "R2_adj": result.rsquared_adj,
            "p_alpha": result.pvalues["const"],
        }

        # Factor betas
        for factor in factors:
            row[factor] = result.params[factor]

        # Factor p-values
        for factor in factors:
            row[f"p_{factor}"] = result.pvalues[factor]

        rows.append(row)

    return pd.DataFrame(rows)


# Compute factor-based expected returns
def compute_factor_expected_returns(data, regression_table, factors=FF6_FACTORS, annualize=True, frequency=FREQUENCY):
    """
    Computes factor-based expected returns using average factor premia.

    Formula:
        E[R_i] = RF + beta_i' E[F]

    Parameters
    ----------
    annualize : bool
        If True, converts monthly expected returns into annual returns.
    """
    factor_means = data[factors + ["RF"]].mean()
    betas = regression_table.set_index("Asset")[factors]
    mu_factor = factor_means["RF"] + betas.dot(factor_means[factors])
    if annualize:
        mu_factor = (1 + mu_factor) ** frequency - 1
    return mu_factor


# Compute naive expected returns
def compute_naive_expected_returns(data, tickers, annualize=True, frequency=FREQUENCY):
    """
    Computes historical average returns from asset returns.
    """
    mu_naive = data[tickers].mean()
    if annualize:
        mu_naive = (1 + mu_naive) ** frequency - 1
    return mu_naive


# Compare naive and factor-based expected returns
def compare_expected_returns(mu_naive, mu_factor):
    comparison = pd.DataFrame({
        "Naive": mu_naive,
        "Factor-based": mu_factor
    })

    comparison["Difference (Factor - Naive)"] = (
        comparison["Factor-based"] - comparison["Naive"]
    )

    return comparison


# Monthly Fama-French risk-free rate, with its full-sample mean
def plot_risk_free_rate(data, figsize=(8, 4)):
    rf = data["RF"].copy()
    rf.index = rf.index.to_timestamp()

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.plot(rf.index, rf * 100, linewidth=1.5, marker="+")

    rf_mean = rf.mean()
    ax.axhline(rf_mean * 100, linestyle="--", label=f"Mean: {rf_mean*100:.2f}%")

    ax.set_title("Fama-French Risk-Free Rate (Monthly)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Risk-Free Rate (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    hide_spines(ax)

    plt.tight_layout()
    return fig


def generate_all_figures():
    set_plot_style()
    data = load_or_fetch_ff6_factors(
        start=START_DATE,
        end=END_DATE,
        snapshot_path=FF6_FACTORS_SNAPSHOT_PATH
    )

    fig = plot_risk_free_rate(data)
    fig.savefig("figures/rf_plot.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved: figures/rf_plot.png")


if __name__ == "__main__":
    generate_all_figures()