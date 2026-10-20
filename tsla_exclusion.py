from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from matplotlib import ticker
from matplotlib.ticker import PercentFormatter

from config import START_DATE, END_DATE, TICKERS, TSLA_DAILY_PRICES_SNAPSHOT_PATH, TSLA_PE_SNAPSHOT_PATH
from data_utils import compute_returns, load_or_fetch_prices
from visualization import set_plot_style, hide_spines

set_plot_style()


# Hides the top/right spines only, keeping the plot "open" on the left/bottom -
# the specific look used throughout this module's charts
def _hide_top_right_spines(ax):
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def _download_daily_prices():
    return load_or_fetch_prices(
        tickers=TICKERS + ["TSLA"],
        start=START_DATE,
        end=END_DATE,
        interval="1d",
        snapshot_path=TSLA_DAILY_PRICES_SNAPSHOT_PATH
    )


# Frozen P/E snapshot (load if present, else fetch live and freeze)
def load_or_fetch_pe_snapshot(pe_benchmark_tickers, snapshot_path):
    """
    Loads a frozen trailing/forward P/E snapshot from `snapshot_path` if it exists,
    otherwise fetches one live via `yf.Ticker(...).info` and freezes it.

    `.info` is always a live, point-in-time snapshot with no historical endpoint, so
    without freezing, every notebook/script re-run silently pulls a different P/E and
    shifts the TSLA valuation comparison. Delete `snapshot_path` to force a fresh pull.
    """

    if snapshot_path.exists():
        snapshot = pd.read_csv(snapshot_path, comment="#", index_col="Label")
        with open(snapshot_path) as f:
            header = f.readline()
        print(f"P/E snapshot loaded from frozen file ({header.lstrip('# ').strip()})")
        return snapshot["P/E"], header.lstrip("# ").strip()

    pe_values = {}
    for label, pe_ticker in pe_benchmark_tickers.items():
        info = yf.Ticker(pe_ticker).info
        pe_values[label] = info.get("trailingPE") or info.get("forwardPE")

    pe_series = pd.Series(pe_values, name="P/E")
    snapshot_date = datetime.today().strftime("%Y-%m-%d")

    with open(snapshot_path, "w") as f:
        f.write(f"# P/E snapshot retrieved on {snapshot_date}\n")
        pe_series.rename_axis("Label").reset_index().to_csv(f, index=False)

    print(f"P/E snapshot fetched live and frozen: {snapshot_date}")
    return pe_series, f"P/E snapshot retrieved on {snapshot_date}"


def plot_realized_vol_ranking(daily_prices, filename="figures/tsla_realized_vol_ranking.png"):
    daily_log_returns = compute_returns(prices=daily_prices, method="log")
    rolling_vol = daily_log_returns.rolling(30).std() * np.sqrt(252)
    median_vol = rolling_vol[TICKERS + ["TSLA"]].median().sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=150)

    bar_colors = ["#C44E52" if ticker_ == "TSLA" else "#4C72B0" for ticker_ in median_vol.index]
    bars = ax.bar(median_vol.index, median_vol.values, width=0.6, color=bar_colors)

    ax.set_ylabel("Median 30-Day Realized Volatility")
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_title("Median Realized Volatility (2015-2026): TSLA vs Investable Universe", fontweight="bold", pad=15)
    ax.bar_label(bars, fmt=lambda v: f"{v:.1%}", padding=3, fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    _hide_top_right_spines(ax)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_realized_vol_rank_over_time(daily_prices, filename="figures/tsla_vol_rank_over_time.png"):
    daily_log_returns = compute_returns(prices=daily_prices, method="log")
    rolling_vol = daily_log_returns.rolling(30).std() * np.sqrt(252)
    rolling_vol = rolling_vol.dropna()

    ranks = rolling_vol[TICKERS + ["TSLA"]].rank(axis=1, ascending=False)
    tsla_rank = ranks["TSLA"]

    fig, ax = plt.subplots(figsize=(10, 4), dpi=150)

    ax.fill_between(tsla_rank.index, 1, tsla_rank.values, color="#C44E52", alpha=0.35, linewidth=0)
    ax.plot(tsla_rank.index, tsla_rank.values, color="#C44E52", linewidth=0.8)

    n_assets = len(TICKERS) + 1
    median_rank = (n_assets + 1) / 2
    ax.axhline(median_rank, color="#555555", linestyle="--", linewidth=1)
    ax.text(tsla_rank.index[15], median_rank, " Universe median rank", color="#555555", fontsize=9, va="bottom")

    ax.set_ylim(n_assets + 0.5, 0.5)
    ax.set_yticks([1, 5, 10, 15, 20])
    ax.set_ylabel("TSLA rank by 30-day realized volatility\n(1 = most volatile of the 20)")
    ax.set_title("Daily Rank of TSLA's Realized Volatility within the Universe", fontweight="bold", pad=15)
    ax.grid(alpha=0.3)

    _hide_top_right_spines(ax)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_normalized_price_vs_universe(daily_prices, filename="figures/tsla_normalized_price_vs_universe.png"):
    monthly_prices = daily_prices.resample("ME").last()
    normalized = monthly_prices / monthly_prices.iloc[0] * 100
    universe_median = normalized[TICKERS].median(axis=1)

    fig, ax = plt.subplots(figsize=(8, 4), dpi=150)

    ax.plot(
        universe_median.index,
        universe_median,
        color="#4C72B0",
        linewidth=2,
        label="Universe median (19 assets)"
    )

    ax.plot(
        normalized.index,
        normalized["TSLA"],
        color="#C44E52",
        linewidth=2.2,
        label="TSLA"
    )

    # (text, offset in points) tuned per event so the label clears the TSLA
    # curve locally instead of using one fixed offset for every point
    events = {
        "2018-08-01": ("\"Funding secured\" tweet", (0, 70)),
        "2020-12-01": ("+700% in 2020", (-110, 50)),
        "2022-12-01": ("-65% in 2022", (60, 70)),
    }

    annotation_box = dict(boxstyle="round,pad=0.3", fc="white", ec="#AAAAAA", alpha=0.9)

    for date_str, (text, offset) in events.items():
        event_date = pd.Timestamp(date_str)
        nearest_pos = normalized.index.get_indexer([event_date], method="nearest")[0]
        nearest_date = normalized.index[nearest_pos]
        y_value = normalized["TSLA"].iloc[nearest_pos]

        ax.annotate(
            text,
            xy=(nearest_date, y_value),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            bbox=annotation_box,
            arrowprops=dict(arrowstyle="->", color="black", lw=0.8)
        )

    ax.set_title("Normalized Price (base 100, Jan. 2015): TSLA vs Universe Median", fontweight="bold", pad=15)
    ax.set_ylabel("Indexed Price (base = 100)")
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", frameon=False)

    _hide_top_right_spines(ax)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_pe_valuation_snapshot(filename="figures/tsla_pe_vs_benchmarks.png"):
    pe_benchmark_tickers = {
        "TSLA": "TSLA",
        "S&P 500 (SPY)": "SPY",
        "Toyota (TM)": "TM",
        "General Motors (GM)": "GM",
        "Ford (F)": "F",
    }

    pe_values, snapshot_label = load_or_fetch_pe_snapshot(pe_benchmark_tickers, TSLA_PE_SNAPSHOT_PATH)
    pe_series = pe_values.sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(6, 4), dpi=150)

    bar_colors = ["#C44E52" if "TSLA" in label else "#4C72B0" for label in pe_series.index]
    bars = ax.bar(pe_series.index, pe_series.values, width=0.5, color=bar_colors)

    ax.set_yscale("log")
    ax.set_ylabel("Price / Earnings (log scale)")
    fig.suptitle("TSLA Valuation vs S&P 500 and Traditional Automakers", fontweight="bold", y=1.02)
    ax.set_title(f"{snapshot_label} (yfinance trailing/forward P/E)",
                 fontsize=8, color="#666666", style="italic", pad=10)
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.grid(axis="y", which="both", linestyle="--", alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")

    _hide_top_right_spines(ax)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return snapshot_label


def plot_debt_equity_ratio(filename="figures/tsla_debt_equity_ratio.png"):
    # Quarterly Debt/Equity data 
    # Source: MacroTrends, cross-checked against SEC 10-K / 10-Q filings
    de_ratio_data = [
        ("2015-03-31", 6.41), ("2015-06-30", 8.04), ("2015-09-30", 4.74), ("2015-12-31", 6.45),
        ("2016-03-31", 8.47), ("2016-06-30", 3.71), ("2016-09-30", 3.70), ("2016-12-31", 3.09),
        ("2017-03-31", 3.32), ("2017-06-30", 3.19), ("2017-09-30", 3.87), ("2017-12-31", 4.47),
        ("2018-03-31", 4.13), ("2018-06-30", 4.90), ("2018-09-30", 4.52), ("2018-12-31", 4.17),
        ("2019-03-31", 4.29), ("2019-06-30", 3.85), ("2019-09-30", 3.77), ("2019-12-31", 3.60),
        ("2020-03-31", 2.71), ("2020-06-30", 2.56), ("2020-09-30", 1.71), ("2020-12-31", 1.26),
        ("2021-03-31", 1.22), ("2021-06-30", 1.15), ("2021-09-30", 1.07), ("2021-12-31", 1.00),
        ("2022-03-31", 0.89), ("2022-06-30", 0.84), ("2022-09-30", 0.83), ("2022-12-31", 0.81),
        ("2023-03-31", 0.78), ("2023-06-30", 0.75), ("2023-09-30", 0.73), ("2023-12-31", 0.68),
        ("2024-03-31", 0.68), ("2024-06-30", 0.68), ("2024-09-30", 0.70), ("2024-12-31", 0.66),
        ("2025-03-31", 0.66), ("2025-06-30", 0.65), ("2025-09-30", 0.66), ("2025-12-31", 0.66),
        ("2026-03-31", 0.70),
    ]

    dates = [datetime.strptime(d, "%Y-%m-%d") for d, _ in de_ratio_data]
    ratios = [r for _, r in de_ratio_data]

    events = [
        ("2018-08-07", 4.52, '"Funding secured"\n(SEC fraud tweet)', "left"),
        ("2020-06-30", 2.56, "Sustained\nprofitability\nbegins", "right"),
        ("2021-10-01", 1.07, "S&P 500 inclusion\n(Dec. 2020) +\nequity raise", "right"),
        ("2022-11-01", 0.83, "Twitter\nacquisition:\nTSLA stock\n-65%", "right"),
    ]

    # Leverage zones: (lower bound, upper bound, color, label)
    leverage_zones = [
        (3.0, 9.5, "#D62728", "Extreme leverage\n(D/E > 3x)"),
        (1.5, 3.0, "#FF7F0E", "High leverage\n(1.5x - 3x)"),
        (0.5, 1.5, "#FFDD57", "Moderate leverage\n(0.5x - 1.5x)"),
        (0.0, 0.5, "#2CA02C", "Low leverage\n(< 0.5x)"),
    ]

    fig, (ax, cax) = plt.subplots(
        1, 2, figsize=(13, 6), dpi=150,
        gridspec_kw={"width_ratios": [10, 1.1], "wspace": 0.04}
    )
    fig.patch.set_facecolor("#FAFAFA")
    ax.set_facecolor("#FAFAFA")
    cax.set_facecolor("#FAFAFA")

    for y0, y1, col, _ in leverage_zones:
        ax.axhspan(y0, y1, alpha=0.10, color=col, zorder=0)

    ax.plot(dates, ratios, color="#1F4E79", linewidth=2.2, zorder=3)
    ax.fill_between(dates, ratios, alpha=0.12, color="#1F4E79", zorder=2)
    ax.scatter(dates, ratios, color="#1F4E79", s=18, zorder=4)

    annotation_style = dict(
        fontsize=7.8, color="#222222",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#AAAAAA", alpha=0.88),
        arrowprops=dict(arrowstyle="-|>", color="#555555", lw=1.0),
    )

    offsets = {
        '"Funding secured"\n(SEC fraud tweet)':         (datetime(2017, 6, 1), 6.0),
        "Sustained\nprofitability\nbegins":              (datetime(2019, 3, 1), 2.0),
        "S&P 500 inclusion\n(Dec. 2020) +\nequity raise": (datetime(2022, 3, 1), 1.8),
        "Twitter\nacquisition:\nTSLA stock\n-65%":       (datetime(2023, 8, 1), 1.6),
    }

    for date_str, y_val, label, _ in events:
        x_pt = datetime.strptime(date_str, "%Y-%m-%d")
        x_txt, y_txt = offsets[label]
        ax.annotate(label, xy=(x_pt, y_val), xytext=(x_txt, y_txt), **annotation_style)

    ax.axhline(1.0, color="#888888", linestyle="--", linewidth=0.9, zorder=1)
    ax.text(datetime(2015, 4, 1), 1.03, "D/E = 1x", color="#888888", fontsize=8)

    ax.set_xlim(datetime(2015, 1, 1), datetime(2026, 6, 1))
    ax.set_ylim(0, 9.5)
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.1f}x"))
    ax.set_xlabel("Date (quarterly data)", labelpad=8)
    ax.set_ylabel("Debt / Equity Ratio (D/E)", labelpad=10)
    ax.set_title("Tesla, Inc. (TSLA) — Debt/Equity Ratio Evolution (2015-2026)", fontsize=11, pad=14)

    fig.text(0.5, 0.94, "Source: MacroTrends / SEC 10-K & 10-Q filings",
              ha="center", fontsize=8, color="#666666", style="italic")

    ax.grid(axis="y", linestyle=":", linewidth=0.5, color="#CCCCCC")
    ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%Y"))

    _hide_top_right_spines(ax)

    cax.set_ylim(ax.get_ylim())
    cax.set_xlim(0, 1)

    for y0, y1, col, lbl in leverage_zones:
        cax.axhspan(y0, y1, color=col, alpha=0.55, zorder=0)
        cax.text(0.5, (y0 + y1) / 2, lbl, ha="center", va="center",
                  fontsize=7.3, fontweight="bold", color="#2A2A2A", linespacing=1.3)

    cax.set_xticks([])
    cax.set_yticks([])
    hide_spines(cax)

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    plt.savefig(filename, dpi=300, bbox_inches="tight", facecolor="#FAFAFA")
    plt.close(fig)


def generate_all_figures():
    daily_prices = _download_daily_prices()
    plot_realized_vol_ranking(daily_prices)
    plot_realized_vol_rank_over_time(daily_prices)
    plot_normalized_price_vs_universe(daily_prices)
    plot_pe_valuation_snapshot()
    plot_debt_equity_ratio()
    print("Saved: tsla_realized_vol_ranking.png, "
          "tsla_vol_rank_over_time.png, tsla_normalized_price_vs_universe.png, "
          "tsla_pe_vs_benchmarks.png, tsla_debt_equity_ratio.png")


if __name__ == "__main__":
    generate_all_figures()