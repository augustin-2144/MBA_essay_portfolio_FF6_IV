"""
Weekly real-money P&L tracking for the ESG-constrained real portfolio. Run once to establish the
purchase (fixed share count, at live prices), then re-run weekly to mark the same fixed position
to market. Target weights come from the thesis's frozen pipeline (FF6 mu, hybrid Sigma at
lambda=0.25, ESG >= 3.4) and do not change week to week; only the live mark-to-market price does.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    TICKERS, START_DATE, END_DATE, FREQUENCY, TARGET_DAYS,
    PRICES_SNAPSHOT_PATH, FF6_FACTORS_SNAPSHOT_PATH, IMPLIED_VOLS_SNAPSHOT_PATH,
    ESG_DATA_PATH, ESG_SHEET_NAME, ESG_SCORE_COLUMN, AMOUNT_INVESTED
)
from data_utils import load_or_fetch_prices, compute_returns, align_data
from Fama_French import load_or_fetch_ff6_factors, build_ff6_regression_table, compute_factor_expected_returns
from implied_volatility import load_or_fetch_implied_vols
from covariance import compute_shrunk_implied_covariance
from esg_scores import load_lseg_esg_scores
from portfolio_builders import build_optimized_portfolio
from visualization import plot_combined_metrics_table
import matplotlib.pyplot as plt

ESG_THRESHOLD = 3.4
LAMBDA = 0.25
BASE_DIR = Path(__file__).resolve().parent
POSITIONS_PATH = BASE_DIR / "data" / "esg_tracking_positions.csv"
LOG_PATH = BASE_DIR / "data" / "esg_tracking_log.csv"
POSITION_TABLE_FIGURE_PATH = BASE_DIR / "figures" / "esg_realmoney_position_table.png"


# Target weights from the frozen thesis pipeline (FF6 mu, hybrid Sigma at lambda=0.25, ESG >= 3.4)
def compute_target_weights():
    prices = load_or_fetch_prices(TICKERS, START_DATE, END_DATE, PRICES_SNAPSHOT_PATH)
    returns = compute_returns(prices)
    factors = load_or_fetch_ff6_factors(START_DATE, END_DATE, FF6_FACTORS_SNAPSHOT_PATH)
    data = align_data(returns, factors)

    risk_free_rate = (1 + data["RF"].mean()) ** FREQUENCY - 1
    regression_table = build_ff6_regression_table(data=data, tickers=TICKERS)
    mu_factor = compute_factor_expected_returns(data=data, regression_table=regression_table, annualize=True)

    implied_vols = load_or_fetch_implied_vols(TICKERS, risk_free_rate, TARGET_DAYS, IMPLIED_VOLS_SNAPSHOT_PATH)
    hist_vols = data[TICKERS].std() * np.sqrt(FREQUENCY)
    implied_vols_filled = implied_vols.fillna(hist_vols)
    Sigma_lambda = compute_shrunk_implied_covariance(returns=data[TICKERS], implied_vols=implied_vols_filled, lambda_=LAMBDA)

    esg_scores = load_lseg_esg_scores(
        path=ESG_DATA_PATH, tickers=TICKERS, score_column=ESG_SCORE_COLUMN, sheet_name=ESG_SHEET_NAME
    )
    bounds = [(0, 0.30)] * len(TICKERS)

    weights = build_optimized_portfolio(
        tickers=TICKERS, mu=mu_factor, Sigma=Sigma_lambda, risk_free_rate=risk_free_rate,
        name=f"ESG >= {ESG_THRESHOLD}", bounds=bounds, esg_scores=esg_scores, min_esg_score=ESG_THRESHOLD
    )
    return weights


# Live spot price for every ticker (today's close, or latest available intraday print)
def get_live_prices():
    data = yf.download(TICKERS, period="1d", interval="1d", auto_adjust=True, progress=False)["Close"]
    prices = data.iloc[-1]
    as_of = data.index[-1].date()
    return prices, as_of


# Styled table figure of the position (same visual convention as the rest of the memoire's figures)
def plot_position_table(positions, amount_invested):
    positions = positions.sort_values("Invested", ascending=False)

    display_table = pd.DataFrame({
        "Weight": [f"{w * 100:.2f}%" for w in positions["Weight"]],
        "Price": [f"${p:,.2f}" for p in positions["Purchase Price"]],
        "Shares": [f"{int(s)}" for s in positions["Shares"]],
        "Invested": [f"${a:,.2f}" for a in positions["Invested"]]
    }, index=positions["Ticker"])

    fig = plot_combined_metrics_table(
        display_table,
        table_groups={"Position": ["Weight", "Price", "Shares", "Invested"]},
        title=f"Real Portfolio -- ESG >= {ESG_THRESHOLD} (${amount_invested:,})",
        index_label="Asset"
    )
    fig.savefig(POSITION_TABLE_FIGURE_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Position table figure saved to {POSITION_TABLE_FIGURE_PATH}")


# Establish the fixed share position (run once)
def establish_position():
    weights = compute_target_weights()
    prices, as_of = get_live_prices()

    dollar_target = weights * AMOUNT_INVESTED
    shares = np.floor(dollar_target / prices).astype(int)
    invested = shares * prices

    positions = pd.DataFrame({
        "Ticker": TICKERS,
        "Weight": weights.values,
        "Purchase Price": prices.reindex(TICKERS).values,
        "Shares": shares.reindex(TICKERS).values,
    })
    positions = positions[positions["Shares"] > 0].reset_index(drop=True)
    positions["Invested"] = positions["Shares"] * positions["Purchase Price"]
    positions.to_csv(POSITIONS_PATH, index=False)

    total_invested = invested.sum()
    cash_leftover = AMOUNT_INVESTED - total_invested

    log_row = pd.DataFrame([{
        "Date": as_of,
        "Portfolio Value": total_invested,
        "Cash Leftover": cash_leftover,
        "Cumulative Return (%)": 0.0,
        "Cumulative PnL ($)": 0.0
    }])
    log_row.to_csv(LOG_PATH, index=False)

    plot_position_table(positions, AMOUNT_INVESTED)

    print(f"Position established on {as_of}: {len(positions)} tickers, ${total_invested:,.2f} invested, "
          f"${cash_leftover:,.2f} cash leftover.")
    print(positions)
    return positions


# Mark the existing fixed position to market and append a log row (run weekly)
def mark_to_market():
    positions = pd.read_csv(POSITIONS_PATH)
    log = pd.read_csv(LOG_PATH)

    baseline_value = float(log.iloc[0]["Portfolio Value"])
    cash_leftover = float(log.iloc[0]["Cash Leftover"])

    prices, as_of = get_live_prices()
    prices = prices.reindex(positions["Ticker"])

    current_value = float((positions["Shares"].values * prices.values).sum())
    cumulative_return = (current_value - baseline_value) / baseline_value * 100
    cumulative_pnl = current_value - baseline_value

    if str(log.iloc[-1]["Date"]) == str(as_of):
        print(f"Already logged for {as_of}, skipping.")
        return log

    new_row = pd.DataFrame([{
        "Date": as_of,
        "Portfolio Value": current_value,
        "Cash Leftover": cash_leftover,
        "Cumulative Return (%)": cumulative_return,
        "Cumulative PnL ($)": cumulative_pnl
    }])
    log = pd.concat([log, new_row], ignore_index=True)
    log.to_csv(LOG_PATH, index=False)

    print(f"Marked to market on {as_of}: value=${current_value:,.2f}, "
          f"cumulative return={cumulative_return:+.2f}%, PnL=${cumulative_pnl:+,.2f}")
    return log


# Entry point
if __name__ == "__main__":
    if not POSITIONS_PATH.exists():
        establish_position()
    else:
        mark_to_market()