"""
Plotting and animation utilities for the thesis's figures: static matplotlib/seaborn
charts, styled tables, and FuncAnimation-based mp4 animations for the defense slides.
Every function returns its Figure (or animation object) rather than saving to disk,
except where noted; the thesis's own build scripts handle saving each figure to the
path referenced from the .tex source.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

from matplotlib.ticker import PercentFormatter
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Rectangle

def set_plot_style():
    """
    Applies the shared matplotlib style (serif/STIX fonts, title and label sizes, grid style)
    used by every figure in this module.
    """

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",

        "axes.titlesize": 18,
        "axes.labelsize": 13,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
        "figure.titlesize": 20,

        "axes.titleweight": "bold",

        "axes.grid": False,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",

        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def hide_spines(ax):
    """
    Hides all four axis spines, used throughout this module for a plain, borderless look.
    """

    for spine in ax.spines.values():
        spine.set_visible(False)

def plot_correlation_matrix(corr_matrix,
                            title="Correlation Matrix",
                            annot=True,
                            mask_upper=True,
                            show=True):
    """
    Heatmap of a correlation matrix, masking the upper triangle by default. Also saves the
    figure to correlation_matrix.png.
    """

    fig, ax = plt.subplots(figsize=(14, 10), dpi=300)
    mask = None

    if mask_upper:
        mask = np.triu(
            np.ones_like(corr_matrix, dtype=bool),
            k=1
        )

    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        annot=annot,
        fmt=".3f",
        annot_kws={"size":8},
        linewidths=0.5,
        linecolor="white",
        square=True,
        cbar_kws={"shrink":0.85,"label":"Correlation"},
        ax=ax
    )

    ax.set_title(
        title,
        fontsize=22,
        weight="bold",
        pad=20
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        fontsize=10
    )

    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        fontsize=10
    )

    ax.set_xlabel("")
    ax.set_ylabel("")

    hide_spines(ax)

    plt.savefig("correlation_matrix.png",dpi=300,bbox_inches="tight")
    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_weights(weights,title="Portfolio Weights",figsize=(12,6)):
    """
    Bar chart of portfolio weights, sorted ascending, with each bar labeled by its percentage
    value.
    """

    weights = pd.Series(weights)

    plot_data = (weights.sort_values().reset_index())
    plot_data.columns = ["Asset", "Weight"]

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    sns.barplot(
        data=plot_data,
        x="Asset",
        y="Weight",
        hue="Asset",
        legend=False,
        ax=ax
    )

    # ===== Adding labels =====
    for p in ax.patches:
        height = p.get_height()
        is_short = height < 0
        va = "top" if is_short else "bottom"
        offset = -3 if is_short else 3

        ax.annotate(
            f"{100*height:.2f}%",
            (p.get_x() + p.get_width()/2,height),
            ha="center",
            va=va,
            fontsize=11,
            xytext=(0, offset),
            textcoords="offset points"
        )

    ax.set_title(
        title,
        fontsize=18,
        weight="bold",
        pad=15
    )

    ax.set_xlabel("")
    ax.set_ylabel("Weight")
    ax.grid(axis="y",linestyle="--",alpha=0.4)

    # extra spacing for the labels
    ax.set_ylim(
        plot_data["Weight"].min() * 1.15,
        plot_data["Weight"].max() * 1.15
    )
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    plt.xticks(rotation=45)

    hide_spines(ax)

    plt.tight_layout()
    return fig,ax


def plot_weight_comparison(weights_table,figsize=(14,7)):
    """
    Compares portfolio weights across several portfolios.
    """

    n_assets = len(weights_table.index)
    n_series = len(weights_table.columns)

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    x = np.arange(n_assets)
    group_width = 0.8
    slot_width = group_width / n_series
    bar_width = slot_width * 0.9
    offsets = (np.arange(n_series) - (n_series - 1) / 2) * slot_width

    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for i, col in enumerate(weights_table.columns):
        color = colors[i % len(colors)]
        positions = x + offsets[i]
        values = weights_table[col].values
        ax.bar(positions, values, width=bar_width, label=col, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(weights_table.index, rotation=0)

    ax.set_title(
        "Portfolio Weight Comparison",
        fontsize=20,
        weight="bold",
        pad=15
    )

    ax.set_ylabel("Weight")
    ax.grid(axis="y",linestyle="--",alpha=0.4)
    ax.legend(frameon=False,fontsize=10)

    hide_spines(ax)

    plt.tight_layout()
    return fig, ax


def plot_risk_contributions(rc_table,title="Risk Contributions",show=True):
    """
    Bar chart of each asset's risk contribution (%), sorted ascending.
    """

    rc = rc_table["Risk Contribution (%)"] * 100
    fig, ax = plt.subplots(figsize=(12,6), dpi=300)
    rc.sort_values().plot(kind="bar",ax=ax)
    ax.set_title(
        title,
        fontsize=18,
        weight="bold",
        pad=15
    )

    ax.set_ylabel("Contribution (%)")
    ax.grid(axis="y",linestyle="--",alpha=0.4)

    hide_spines(ax)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax

def plot_efficient_frontier(frontier_summaries,
                            title="Efficient Frontier",
                            figsize=(11, 7),
                            legend_loc="upper left",
                            legend_bbox_to_anchor=(1.02, 1)):
    """
    Plots one or several efficient frontiers on the same chart.

    Parameters
    ----------
    frontier_summaries : dict
        Mapping from a label (e.g. "lambda = 0.00") to the dict returned
        by `compute_frontier_summary` (must contain "frontier",
        "tangent_return", "tangent_volatility", "min_var_return" and
        "min_var_volatility").
    legend_loc, legend_bbox_to_anchor
        Forwarded to `ax.legend`. Default places the legend outside the
        axes on the right; pass e.g. legend_loc="upper left",
        legend_bbox_to_anchor=(0.60, 0.62) to dock it inside an empty
        region of the plot instead.
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    colors = sns.color_palette("husl", len(frontier_summaries))

    for color, (label, summary) in zip(colors, frontier_summaries.items()):
        frontier = summary["frontier"]

        ax.plot(
            frontier["Volatility"] * 100,
            frontier["Return"] * 100,
            color=color,
            linewidth=2,
            label=label
        )

        ax.scatter(
            summary["tangent_volatility"] * 100,
            summary["tangent_return"] * 100,
            color=color,
            marker="*",
            s=300,
            edgecolor="black",
            linewidth=0.8,
            zorder=5
        )

        ax.scatter(
            summary["min_var_volatility"] * 100,
            summary["min_var_return"] * 100,
            color=color,
            marker="D",
            s=70,
            edgecolor="black",
            linewidth=0.8,
            zorder=5
        )

    ax.set_xlabel("Volatility (%)")
    ax.set_ylabel("Expected Return (%)")
    ax.set_title(title, fontweight="bold")

    ax.grid(alpha=0.3)

    handles, labels = ax.get_legend_handles_labels()
    star_marker = plt.Line2D([], [], color="black", marker="*", linestyle="None", markersize=14, label="Max Sharpe (tangency)")
    diamond_marker = plt.Line2D([], [], color="black", marker="D", linestyle="None", markersize=8, label="Min variance")
    ax.legend(
        handles=handles + [star_marker, diamond_marker],
        fontsize=10,
        loc=legend_loc,
        bbox_to_anchor=legend_bbox_to_anchor
    )
    plt.tight_layout()
    return fig, ax

def plot_factor_exposures(regression_table, factors=("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom")):
    """
    Renders FF6 betas and alpha as a styled table: color intensity scales
    with coefficient magnitude, and significance stars come from each
    coefficient's p-value (alpha is only colored when significant at 10%).
    """

    factors = list(factors)
    cols = ["alpha"] + factors
    pretty_cols = ["Asset", r"$\alpha$ (%/month)"] + factors + [r"Adj. $R^2$"]

    max_abs = {col: regression_table[col].abs().max() for col in cols}

    def stars(p):
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        if p < 0.10:
            return "·"
        return ""

    def blend(color, intensity):
        white = np.array(mcolors.to_rgb("white"))
        base = np.array(mcolors.to_rgb(color))
        mixed = white * (1 - intensity) + base * intensity
        return mcolors.to_hex(mixed)

    def cell_color(value, col, p=None):
        if col == "alpha":
            if p is not None and p < 0.10:
                intensity = 0 if max_abs[col] == 0 else min(abs(value) / max_abs[col], 1.0)
                color = "#6BBF59" if value > 0 else "#D66060"
                return blend(color, 0.35 + 0.55 * intensity)
            return "#F3F4F6"

        if col == "R2_adj":
            return blend("#4C72B0", 0.20 + 0.45 * value)

        if value == 0:
            return "#F3F4F6"
        intensity = 0 if max_abs[col] == 0 else min(abs(value) / max_abs[col], 1.0)
        color = "#4C72B0" if value > 0 else "#D66060"
        return blend(color, 0.20 + 0.60 * intensity)

    table_data = []
    for _, row in regression_table.iterrows():
        cells = [row["Asset"], f"{row['alpha']:+.2f}{stars(row['p_alpha'])}"]
        cells += [f"{row[f]:+.2f}{stars(row[f'p_{f}'])}" for f in factors]
        cells.append(f"{row['R2_adj']:.3f}")
        table_data.append(cells)

    n_assets = len(table_data)
    n_cols = len(pretty_cols)

    # Row height is fixed in inches so the table stays readable regardless
    # of how many assets are in the universe, instead of squeezing every
    # row into one fixed-size figure.
    row_h_in, header_h_in = 0.42, 0.55
    title_h_in, legend_h_in = 1.1, 0.85
    table_h_in = header_h_in + row_h_in * n_assets
    fig_width = 16

    fig = plt.figure(figsize=(fig_width, title_h_in + table_h_in + legend_h_in), dpi=300)
    gs = fig.add_gridspec(
        3, 1,
        height_ratios=[title_h_in, table_h_in, legend_h_in],
        hspace=0
    )

    ax_title = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])
    ax_legend = fig.add_subplot(gs[2])
    for ax in (ax_title, ax_table, ax_legend):
        ax.axis("off")

    table = ax_table.table(
        cellText=table_data,
        colLabels=pretty_cols,
        cellLoc="center",
        colLoc="center",
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)

    for i in range(n_assets + 1):
        for j in range(n_cols):
            cell = table[i, j]
            cell.set_edgecolor("#C7CCD6")
            cell.set_linewidth(1.0)

    for j in range(n_cols):
        header = table[0, j]
        header.set_facecolor("#243B6B")
        header.set_edgecolor("white")
        header.set_linewidth(1.2)
        header.set_text_props(color="white", weight="bold", fontsize=11.5)

    asset_band_colors = ["#F8F9FB", "#EEF0F4"]

    for i, (_, row) in enumerate(regression_table.iterrows()):
        for j, col in enumerate(["Asset"] + cols + ["R2_adj"]):
            cell = table[i + 1, j]

            if col == "Asset":
                cell.set_facecolor(asset_band_colors[i % 2])
                cell.set_text_props(weight="bold", fontsize=10.5, color="#1F2937")
                continue

            if col == "R2_adj":
                cell.set_facecolor(cell_color(row["R2_adj"], "R2_adj"))
                cell.set_text_props(fontsize=10.5, color="#111827")
                continue

            p = row["p_alpha"] if col == "alpha" else row[f"p_{col}"]
            cell.set_facecolor(cell_color(row[col], col, p))
            cell.set_text_props(fontsize=10.5, color="#111827")

    ax_title.text(
        0.5, 0.78, "Fama-French 6-Factor Regression Results",
        ha="center", va="center", fontsize=17, fontweight="bold", color="#111827",
        transform=ax_title.transAxes
    )
    ax_title.text(
        0.5, 0.42, "Monthly OLS estimation on excess returns",
        ha="center", va="center", fontsize=12, color="#4B5563",
        transform=ax_title.transAxes
    )
    ax_title.text(
        0.5, 0.10,
        r"Significance : $\qquad ^{***}p<0.001 \qquad ^{**}p<0.01 \qquad ^{*}p<0.05 \qquad ^{\cdot}p<0.10$",
        ha="center", va="center", fontsize=12, color="#6B7280",
        transform=ax_title.transAxes
    )

    box_w, box_h = 0.022, 0.16
    legend_y = 0.62
    legend_items = [
        ("#4C72B0", r"$\beta > 0$"),
        ("#D66060", r"$\beta < 0$"),
        ("#6BBF59", r"$\alpha$ significant and positive"),
        ("#F3F4F6", "non-significant / low intensity"),
    ]

    for (color, label), x in zip(legend_items, [0.15, 0.33, 0.54, 0.75]):
        ax_legend.add_patch(Rectangle(
            (x, legend_y), box_w, box_h,
            transform=ax_legend.transAxes,
            facecolor=color, edgecolor="#C7CDD8", linewidth=0.9
        ))
        ax_legend.text(
            x + box_w + 0.01, legend_y + box_h / 2, label,
            ha="left", va="center", fontsize=12, color="#374151",
            transform=ax_legend.transAxes
        )

    ax_legend.text(
        0.5, 0.18,
        "Color intensity is proportional to the magnitude of the estimated coefficients.",
        ha="center", va="center", fontsize=12, color="#6B7280",
        transform=ax_legend.transAxes
    )
    return fig

def plot_factor_exposure_dashboard(regression_table,
                                   beta_cols=("Mkt-RF", "SMB", "HML", "RMW", "CMA", "Mom"),
                                   hml_growth_threshold=-0.25,
                                   hml_significance=0.10,
                                   figsize=(15, 7)):
    """
    Dashboard-style summary of FF6 factor exposures: a row of KPI cards
    (cross-sectional averages / counts) followed by a bar chart of the
    average loading on each factor across the investment universe.

    All KPI values are computed from `regression_table`, not hardcoded.
    """

    beta_cols = list(beta_cols)
    table = regression_table.set_index("Asset")
    beta_table = table[beta_cols]
    means = beta_table.mean()

    # only count an asset as "growth" if its negative HML loading also
    # clears the same significance bar used everywhere else (stars in
    # plot_factor_exposures), so a noisy coefficient can't pad the count
    n_growth = int(((beta_table["HML"] < hml_growth_threshold) & (table["p_HML"] < hml_significance)).sum())
    n_assets = len(beta_table)

    card_color = "#4C72B0"
    card_neg_color = "#C44E52"

    kpis = [
        ("Avg. Market $\\beta$", f"{means['Mkt-RF']:.2f}", "Near-systematic exposure", card_color),
        ("Dominant SMB Bias", f"{means['SMB']:+.2f}", "Large-cap bias", card_neg_color),
        ("Growth Assets (HML-)", f"{n_growth}/{n_assets}", f"HML < {hml_growth_threshold:.2f}, sig. at {hml_significance:.0%}", "#55A868"),
        ("Average Momentum", f"{means['Mom']:+.2f}", "Near-neutral exposure", "#DD8452"),
    ]

    bg_color = "#0d1117"
    card_bg = "#161b22"

    fig = plt.figure(figsize=figsize, dpi=300, facecolor=bg_color)
    gs = fig.add_gridspec(2, len(kpis), height_ratios=[1, 1.6], hspace=0.12, wspace=0.18)

    for col, (label, value, subtitle, color) in enumerate(kpis):
        ax_card = fig.add_subplot(gs[0, col])
        ax_card.set_facecolor(card_bg)
        ax_card.axis("off")

        for spine_pos in ["top", "bottom", "left", "right"]:
            ax_card.spines[spine_pos].set_visible(False)

        ax_card.text(0.08, 0.78, label, color="#8b949e", fontsize=10,
                     fontweight="bold", ha="left", va="center", transform=ax_card.transAxes)
        ax_card.text(0.08, 0.45, value, color=color, fontsize=26,
                     fontweight="bold", ha="left", va="center", transform=ax_card.transAxes)
        ax_card.text(0.08, 0.15, subtitle, color="#c9d1d9", fontsize=10,
                     ha="left", va="center", transform=ax_card.transAxes)

    ax_bar = fig.add_subplot(gs[1, :])
    ax_bar.set_facecolor(bg_color)

    bar_colors = [card_color if v >= 0 else card_neg_color for v in means]

    ax_bar.bar(means.index, means.values, color=bar_colors, width=0.6)
    ax_bar.axhline(0, color="white", linewidth=0.8)

    ax_bar.set_title(
        "Average Exposures by Factor",
        color="white",
        fontsize=15,
        fontweight="bold",
        loc="left",
        pad=15
    )
    ax_bar.tick_params(colors="white")
    ax_bar.set_ylabel("Average Beta", color="white")

    for spine in ax_bar.spines.values():
        spine.set_color("#30363d")

    ax_bar.grid(axis="y", color="#30363d", linestyle="-", linewidth=0.5, alpha=0.6)
    ax_bar.set_axisbelow(True)

    return fig

def plot_expected_return_comparison(expected_return_comparison, show=True):
    """
    Grouped bar chart comparing naive (historical) vs factor-based expected returns, by asset.
    """

    plot_data = (
        expected_return_comparison
        .reset_index()
        .rename(columns={"index":"Asset"})
    )

    plot_data = plot_data.melt(
        id_vars="Asset",
        value_vars=["Naive", "Factor-based"],
        var_name="Method",
        value_name="Expected Return"
    )

    plot_data["Expected Return"] *= 100
    fig, ax = plt.subplots(figsize=(14, 7), dpi=300)
    sns.barplot(
        data=plot_data,
        x="Asset",
        y="Expected Return",
        hue="Method",
        ax=ax
    )

    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        ha="right",
        fontsize=20
    )
    ax.set_title(
        "Expected Return Comparison",
        fontsize=20,
        weight="bold",
        pad=20
    )
    ax.set_xlabel("")
    ax.set_ylabel("Expected Return (%)")
    ax.grid(
        axis="y",
        linestyle="--",
        alpha=0.4
    )
    ax.legend(
        frameon=False
    )

    hide_spines(ax)

    plt.tight_layout()
    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


def plot_combined_metrics_table(metrics_table,
                                table_groups=None,
                                title="Portfolio Metrics Summary",
                                index_label="Portfolio",
                                column_labels=None,
                                zero_highlight_columns=None,
                                figsize=None):
    """
    Styled summary table figure grouping metrics under column headers (e.g. Performance /
    Diversification). table_groups maps each group label to the list of metric columns it
    covers; defaults to the thesis's standard Performance/Diversification split if not provided.
    """

    if table_groups is None:
        table_groups = {
            "Performance Metrics": [
                "Expected Return",
                "Volatility",
                "Sharpe Ratio"
            ],
            "Diversification Metrics": [
                "HHI Normalized",
                "Effective Number of Positions"
            ]
        }

    group_columns = {}
    for group, columns in table_groups.items():
        available_columns = [col for col in columns if col in metrics_table.columns]

        if available_columns:
            group_columns[group] = available_columns

    ordered_columns = [col for cols in group_columns.values() for col in cols]
    display_table = metrics_table[ordered_columns]

    n_rows = len(display_table)
    n_cols = len(ordered_columns) + 1

    show_group_row = len(group_columns) > 1

    group_header_row = [""] * n_cols

    column_labels = column_labels or {}
    column_header_row = [index_label] + [column_labels.get(col, col) for col in ordered_columns]

    data_rows = [
        [str(idx)] + [str(value) for value in row.values]
        for idx, row in display_table.iterrows()
    ]

    cell_text = ([group_header_row] if show_group_row else []) + [column_header_row] + data_rows
    col_char_lengths = [
        float(max(len(str(row[col])) for row in [column_header_row] + data_rows))
        for col in range(n_cols)
    ]
    col_char_lengths = [max(length, 6.0) for length in col_char_lengths]

    total_chars = sum(col_char_lengths)
    col_widths = [length / total_chars for length in col_char_lengths]

    if figsize is None:
        figsize = (total_chars * 0.11 + 1.0, 1.4 + 0.55 * n_rows + (0.4 if show_group_row else 0))

    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.axis("off")

    table = ax.table(
        cellText=cell_text,
        cellLoc="center",
        colWidths=col_widths,
        bbox=[0, 0, 1, 1]
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.4)

    header_color = "#243B6B"
    group_band_color = "#3B5998"
    row_label_color = "#EAEAF2"
    body_color = "#F8F9FA"
    zero_tint = (1.0, 0.0, 0.0, 0.10)

    zero_highlight_columns = set(zero_highlight_columns or [])

    def _is_zero_display(value):
        # mirrors plot_weights_table: a cell counts as "zero" when its
        # displayed percentage would round to 0.0% at 1 decimal
        text = str(value).strip()
        if text.endswith("%"):
            text = text[:-1]
        try:
            return abs(float(text)) < 0.05
        except ValueError:
            return False

    column_header_row_idx = 1 if show_group_row else 0

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("white")

        if show_group_row and row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor(header_color if col == 0 else group_band_color)

        elif row == column_header_row_idx:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor(header_color)

        elif col == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor(row_label_color)

        elif (ordered_columns[col - 1] in zero_highlight_columns
              and _is_zero_display(cell_text[row][col])):
            cell.set_facecolor(zero_tint)

        else:
            cell.set_facecolor(body_color)

    if show_group_row:
        for col in range(n_cols):
            table[(0, col)].set_edgecolor("none")

        fig.canvas.draw()
        row0_cells = [table[(0, col)] for col in range(n_cols)]
        y_top = max(cell.get_y() + cell.get_height() for cell in row0_cells)
        y_bottom = min(cell.get_y() for cell in row0_cells)

        x_edges = [0.0]
        for width in col_widths:
            x_edges.append(x_edges[-1] + width)

        boundary_cols = [0, 1]
        col_cursor = 1
        for cols in group_columns.values():
            col_cursor += len(cols)
            boundary_cols.append(col_cursor)

        line_kwargs = dict(color="white", linewidth=1.2, zorder=5, transform=ax.transAxes)
        ax.plot([0, 1], [y_top, y_top], **line_kwargs)
        ax.plot([0, 1], [y_bottom, y_bottom], **line_kwargs)
        for col in boundary_cols:
            ax.plot([x_edges[col], x_edges[col]], [y_bottom, y_top], **line_kwargs)

        y_center = (y_top + y_bottom) / 2
        renderer = fig.canvas.get_renderer()
        fig_width_px = fig.get_size_inches()[0] * fig.dpi

        for group, (start_col, end_col) in zip(group_columns.keys(), zip(boundary_cols[1:], boundary_cols[2:])):
            x_center = (x_edges[start_col] + x_edges[end_col]) / 2
            span_width_px = (x_edges[end_col] - x_edges[start_col]) * fig_width_px

            label = ax.text(
                x_center, y_center, group,
                ha="center", va="center",
                weight="bold", color="white", fontsize=10,
                zorder=6, transform=ax.transAxes
            )

            # the span's width is set by its own columns' content, not by
            # this label, so if the label doesn't fit, shrink it - never
            # the other way around
            text_width_px = label.get_window_extent(renderer=renderer).width
            max_width_px = span_width_px * 0.92

            if text_width_px > max_width_px > 0:
                label.set_fontsize(max(10 * max_width_px / text_width_px, 7))

    ax.set_title(
        title,
        fontsize=16,
        weight="bold",
        pad=10
    )
    plt.tight_layout()
    return fig

def plot_weights_table(weights_table,
                       title="Portfolio Weights",
                       assets_per_block=None,
                       index_label="Portfolio",
                       figsize=None):
    """
    Compact, LaTeX/Word-export-friendly table of portfolio weights: assets
    as columns, portfolios as rows, split into side-by-side blocks so a long
    asset list stays wide and short rather than producing a tall figure.
    Every weight is shown explicitly as a percentage (including "0.0%" where
    the optimizer truly allocates nothing), with every ~0 cell given a light
    red tint to flag what the optimizer ignored, and the largest non-zero
    weight in each asset's column bolded.
    """

    zero_threshold = 0.0005  # rounds to "0.0%" at the displayed precision
    zero_tint = (1.0, 0.0, 0.0, 0.10)

    transposed = weights_table.T
    assets = list(transposed.columns)
    n_assets = len(assets)
    portfolios = list(transposed.index)
    n_portfolios = len(portfolios)

    if assets_per_block is None:
        assets_per_block = -(-n_assets // 2)  # ceil(n_assets / 2): two blocks by default

    blocks = [
        assets[i:i + assets_per_block]
        for i in range(0, n_assets, assets_per_block)
    ]

    header_color = "#243B6B"
    row_label_color = "#EAEAF2"
    body_color = "#F8F9FA"

    row_height = 0.42
    block_height = row_height * (n_portfolios + 1) + 0.25
    max_block_width = max(len(block) for block in blocks)

    if figsize is None:
        figsize = (
            max(max_block_width * 0.85 + 2.4, 6),
            block_height * len(blocks) + 0.5
        )

    fig, axes = plt.subplots(len(blocks), 1, figsize=figsize, dpi=300)
    if len(blocks) == 1:
        axes = [axes]

    for ax, block_assets in zip(axes, blocks):
        ax.axis("off")

        n_cols = len(block_assets) + 1
        column_header_row = [index_label] + block_assets
        data_rows = [
            [str(portfolio)] + [
                f"{100 * transposed.loc[portfolio, asset]:.1f}%"
                for asset in block_assets
            ]
            for portfolio in portfolios
        ]
        cell_text = [column_header_row] + data_rows

        col_char_lengths = [
            float(max(len(str(row[col])) for row in cell_text))
            for col in range(n_cols)
        ]
        col_char_lengths = [max(length, 6.0) for length in col_char_lengths]
        total_chars = sum(col_char_lengths)
        col_widths = [length / total_chars for length in col_char_lengths]

        # which portfolio (row) carries the largest weight for each asset,
        # used to bold the dominant allocation when it is non-zero
        col_max_row = {
            j: 1 + int(np.argmax([transposed.loc[p, asset] for p in portfolios]))
            for j, asset in enumerate(block_assets)
        }

        table = ax.table(
            cellText=cell_text,
            cellLoc="center",
            colWidths=col_widths,
            bbox=[0, 0, 1, 1]
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 1.4)

        for (row, col), cell in table.get_celld().items():
            cell.set_edgecolor("white")

            if row == 0:
                cell.set_text_props(weight="bold", color="white")
                cell.set_facecolor(header_color)
            elif col == 0:
                cell.set_text_props(weight="bold")
                cell.set_facecolor(row_label_color)
            elif transposed.loc[portfolios[row - 1], block_assets[col - 1]] < zero_threshold:
                cell.set_facecolor(zero_tint)
            else:
                cell.set_facecolor(body_color)
                if row == col_max_row.get(col - 1):
                    cell.set_text_props(weight="bold")

    fig.suptitle(
        title,
        fontsize=16,
        weight="bold"
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    return fig

def plot_portfolio_summary_table(summary,
                                 title="Metrics summary of the portfolio",
                                 figsize=(7, 3.8),
                                 save_path=None):
    """
    Compact two-column table (metric, formatted value) summarizing a single portfolio's key
    metrics, with per-metric formatting (percentages, Sharpe ratio to 3 decimals, etc.).
    Optionally saves the figure to save_path.
    """

    summary_display = summary.copy()
    percent_metrics = [
        "Expected Return",
        "Volatility",
        "Long Exposure",
        "Short Exposure",
        "Gross Exposure",
        "Net Exposure"
    ]

    rows = []

    for metric, value in summary_display.items():

        if metric in percent_metrics:
            formatted_value = f"{value:.2f}%"

        elif metric == "Sharpe Ratio":
            formatted_value = f"{value:.3f}"

        elif metric == "HHI Normalized":
            formatted_value = f"{value:.3f}"

        elif metric == "Effective Number of Positions":
            formatted_value = f"{value:.2f}"

        else:
            formatted_value = f"{value:.3f}"

        rows.append([metric, formatted_value])

    table_data = pd.DataFrame(rows,columns=["Metric", "Value"])
    fig, ax = plt.subplots(figsize=figsize, dpi=300)
    ax.axis("off")

    table = ax.table(
        cellText=table_data.values,
        colLabels=table_data.columns,
        cellLoc="center",
        colLoc="center",
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    for (row, col), cell in table.get_celld().items():

        if row == 0:
            cell.set_text_props(weight="bold", color="white")
            cell.set_facecolor("#243B6B")

        elif col == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#EAEAF2")

        else:
            cell.set_facecolor("#F8F9FA")

        cell.set_edgecolor("white")

    ax.set_title(
        title,
        fontsize=15,
        weight="bold",
        pad=12
    )

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(
            save_path,
            dpi=300,
            bbox_inches="tight"
        )

    return fig, ax


def animate_lambda_weights(lambda_weights,
                           lambda_grid,
                           order_by=None,
                           steps_per_transition=15,
                           fps=36,
                           dpi=150,
                           filename="lambda_weights_animation.mp4",
                           title="Portfolio Allocation as a Function of the Shrinkage Coefficient"):
    """
    Animates portfolio weights as bars that morph smoothly across the
    lambda grid (assets keep a fixed position, only their height changes),
    for use as a presentation slide. Saves an mp4 via ffmpeg and returns
    the animation object.
    """

    lambda_labels = [f"{l:.2f}" for l in lambda_grid]

    assets = list(lambda_weights.index)
    if order_by is not None:
        order_label = order_by if order_by in lambda_weights.columns else f"{float(order_by):.2f}"
        assets = list(lambda_weights[order_label].sort_values(ascending=False).index)

    weights_ordered = lambda_weights.loc[assets, lambda_labels].values

    # interpolate intermediate frames between consecutive lambda steps so
    # the bars morph smoothly instead of jumping every 0.05 step
    frames = []
    frame_lambdas = []
    for i in range(len(lambda_grid) - 1):
        for s in range(steps_per_transition):
            t = s / steps_per_transition
            frames.append(weights_ordered[:, i] * (1 - t) + weights_ordered[:, i + 1] * t)
            frame_lambdas.append(lambda_grid[i] * (1 - t) + lambda_grid[i + 1] * t)
    frames.append(weights_ordered[:, -1])
    frame_lambdas.append(lambda_grid[-1])

    colors = plt.cm.tab20(np.linspace(0, 1, len(assets)))

    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    x = np.arange(len(assets))
    bars = ax.bar(x, frames[0], color=colors)
    for bar in bars:
        bar.set_animated(True)

    ax.set_xticks(x)
    ax.set_xticklabels(assets, rotation=45, ha="right")
    ax.set_ylabel("Portfolio Weight")
    ax.set_ylim(0, max(weights_ordered.max() * 1.15, 0.05))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    hide_spines(ax)

    lambda_text = ax.text(
        0.98, 0.95, "", transform=ax.transAxes,
        ha="right", va="top", fontsize=18, fontweight="bold",
        animated=True
    )

    n_assets = len(assets)
    active_threshold = 1e-6
    active_text = ax.text(
        0.98, 0.87, "", transform=ax.transAxes,
        ha="right", va="top", fontsize=16, fontweight="bold", color="#1E2761",
        animated=True
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout()

    def update(frame_idx):
        heights = frames[frame_idx]
        for bar, height in zip(bars, heights):
            bar.set_height(height)
        lambda_text.set_text(fr"$\lambda$ = {frame_lambdas[frame_idx]:.2f}")
        n_active = int(np.sum(heights > active_threshold))
        active_text.set_text(f"{n_active} / {n_assets} active positions")
        return list(bars) + [lambda_text, active_text]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=True)
    anim.save(
        filename,
        writer="ffmpeg",
        fps=fps,
        dpi=dpi,
        extra_args=["-preset", "veryfast", "-crf", "23"]
    )
    plt.close(fig)
    return anim


def animate_lambda_risk_contributions(rc_grid,
                                       lambda_grid,
                                       assets_order=None,
                                       steps_per_transition=15,
                                       fps=36,
                                       dpi=150,
                                       filename="lambda_risk_contributions_animation.mp4",
                                       title="Risk Contribution as a Function of the Shrinkage Coefficient"):
    """
    Animates each asset's risk contribution (%) as bars that morph smoothly
    across the lambda grid, mirroring animate_lambda_weights. Pass the same
    `assets_order` used for the weights animation (e.g. the ordering derived
    there) so both videos share the same x-axis, for a direct side-by-side
    comparison of weight vs risk.

    `rc_grid` : pd.DataFrame indexed by asset, one column per lambda label
    (same orientation as `lambda_weights`), with risk-contribution fractions
    (summing to 1 down each column).
    """

    lambda_labels = [f"{l:.2f}" for l in lambda_grid]

    assets = assets_order if assets_order is not None else list(rc_grid.index)
    rc_ordered = rc_grid.loc[assets, lambda_labels].values

    frames = []
    frame_lambdas = []
    for i in range(len(lambda_grid) - 1):
        for s in range(steps_per_transition):
            t = s / steps_per_transition
            frames.append(rc_ordered[:, i] * (1 - t) + rc_ordered[:, i + 1] * t)
            frame_lambdas.append(lambda_grid[i] * (1 - t) + lambda_grid[i + 1] * t)
    frames.append(rc_ordered[:, -1])
    frame_lambdas.append(lambda_grid[-1])

    # freeze on, and highlight, the moment the top risk contributor changes
    # hands -- otherwise it slips by in a single frame at 36fps and is easy
    # to miss
    pause_frames = int(round(fps * 1.0))
    top_per_frame = [int(np.argmax(f)) for f in frames]

    paused_frames, paused_lambdas, highlight = [], [], []
    prev_top = top_per_frame[0]
    for f, fl, top in zip(frames, frame_lambdas, top_per_frame):
        if top != prev_top:
            for _ in range(pause_frames):
                paused_frames.append(f)
                paused_lambdas.append(fl)
                highlight.append(top)
            prev_top = top
        paused_frames.append(f)
        paused_lambdas.append(fl)
        highlight.append(None)
    frames, frame_lambdas = paused_frames, paused_lambdas

    colors = plt.cm.tab20(np.linspace(0, 1, len(assets)))

    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)
    x = np.arange(len(assets))
    bars = ax.bar(x, frames[0], color=colors, edgecolor="none", linewidth=0)
    for bar in bars:
        bar.set_animated(True)

    ax.set_xticks(x)
    ax.set_xticklabels(assets, rotation=45, ha="right")
    ax.set_ylabel("Risk Contribution")
    ax.set_ylim(0, max(rc_ordered.max() * 1.15, 0.05))
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    hide_spines(ax)

    lambda_text = ax.text(
        0.98, 0.95, "", transform=ax.transAxes,
        ha="right", va="top", fontsize=18, fontweight="bold",
        animated=True
    )

    top_text = ax.text(
        0.98, 0.87, "", transform=ax.transAxes,
        ha="right", va="top", fontsize=16, fontweight="bold", color="#C99A3D",
        animated=True
    )

    ax.set_title(title, fontsize=16, fontweight="bold", pad=15)
    plt.tight_layout()

    def update(frame_idx):
        heights = frames[frame_idx]
        hl = highlight[frame_idx]
        for j, (bar, height) in enumerate(zip(bars, heights)):
            bar.set_height(height)
            if hl is not None and j == hl:
                bar.set_edgecolor("black")
                bar.set_linewidth(3)
            else:
                bar.set_edgecolor("none")
                bar.set_linewidth(0)
        lambda_text.set_text(fr"$\lambda$ = {frame_lambdas[frame_idx]:.2f}")
        top_idx = int(np.argmax(heights))
        top_text.set_text(f"{assets[top_idx]}: {heights[top_idx] * 100:.1f}% of risk")
        return list(bars) + [lambda_text, top_text]

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=True)
    anim.save(
        filename,
        writer="ffmpeg",
        fps=fps,
        dpi=dpi,
        extra_args=["-preset", "veryfast", "-crf", "23"]
    )
    plt.close(fig)
    return anim


def animate_backtest_cumulative_performance(cum_perf,
                                             fps=10,
                                             dpi=150,
                                             filename="backtest_cumulative_performance_animation.mp4",
                                             title="Rolling Backtest — Cumulative Performance"):
    """
    Animates the rolling-backtest cumulative performance curves month by
    month, each line growing with a live marker at its tip, mirroring
    plot_backtest_cumulative_performance for styling (AI-cycle marker,
    axis labels, spine hiding). Adds a live date readout and a running
    Historical Benchmark vs Factor-Based (FF6) return counter.
    """
    n = len(cum_perf)
    x_labels = cum_perf.index.astype(str)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(12, 6), dpi=dpi)

    lines = {}
    dots = {}
    for column in cum_perf.columns:
        (line,) = ax.plot([], [], linewidth=2, label=column, animated=True)
        (dot,) = ax.plot([], [], marker="o", markersize=7, color=line.get_color(), animated=True)
        lines[column] = line
        dots[column] = dot

    ax.set_xlim(0, n - 1)
    y_max = cum_perf.max().max()
    y_min = min(cum_perf.min().min(), 1.0)
    ax.set_ylim(y_min * 0.95, y_max * 1.08)
    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_ylabel("Growth of $1")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ai_cycle_pos = cum_perf.index.get_indexer([pd.Period("2023-01", freq="M")])[0]
    ax.axvline(ai_cycle_pos - 0.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(
        ai_cycle_pos - 0.5, ax.get_ylim()[1], "  AI cycle start (Jan. 2023)",
        rotation=90, va="top", ha="left", fontsize=9, color="gray"
    )

    ax.legend(frameon=False, loc="upper left")
    hide_spines(ax)

    step = max(n // 15, 1)
    ax.set_xticks(np.arange(n)[::step])
    ax.set_xticklabels(x_labels[::step], rotation=45, ha="right")

    date_text = ax.text(
        0.98, 0.95, "", transform=ax.transAxes,
        ha="right", va="top", fontsize=16, fontweight="bold",
        animated=True
    )

    # running per-series return counter, centered at the top of the plot
    # (clear of the curves, which always end up crowding the right edge),
    # each entry colored to match its own line
    perf_texts = {}
    perf_x = {col: xpos for col, xpos in zip(cum_perf.columns, [0.32, 0.5, 0.68])}
    for column in cum_perf.columns:
        perf_texts[column] = ax.text(
            perf_x[column], 0.90, "", transform=ax.transAxes,
            ha="center", va="top", fontsize=14, fontweight="bold",
            color=lines[column].get_color(), animated=True
        )

    plt.tight_layout()

    short_label = {
        "Historical Benchmark": "Historical",
        "Factor-Based (FF6)": "FF6",
        "S&P 500 (SPY)": "S&P 500",
    }

    def update(frame_idx):
        idx = frame_idx + 1
        artists = []
        for column in cum_perf.columns:
            y = cum_perf[column].values[:idx]
            lines[column].set_data(x[:idx], y)
            dots[column].set_data([x[idx - 1]], [y[-1]])
            ret = (y[-1] - 1) * 100
            label = short_label.get(column, column)
            perf_texts[column].set_text(f"{label} {ret:+.0f}%")
            artists += [lines[column], dots[column], perf_texts[column]]
        date_text.set_text(str(cum_perf.index[idx - 1]))
        return artists + [date_text]

    anim = FuncAnimation(fig, update, frames=n, interval=1000 / fps, blit=True)
    anim.save(
        filename,
        writer="ffmpeg",
        fps=fps,
        dpi=dpi,
        extra_args=["-preset", "veryfast", "-crf", "23"]
    )
    plt.close(fig)
    return anim


def plot_lambda_weights_heatmap(lambda_weights, tickers, figsize=(22, 11)):
    """
    Heatmap of portfolio weights across the full shrinkage grid lambda, one row per lambda
    value.
    """

    df_plot = lambda_weights.copy()
    if df_plot.shape[0] == len(tickers):
        df_plot = df_plot.T

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    fig.patch.set_facecolor("#061826")
    ax.set_facecolor("#061826")

    im = ax.imshow(
        df_plot.values,
        cmap=plt.cm.Blues,
        aspect="auto",
        vmin=0,
        vmax=max(0.40, df_plot.max().max())
    )

    ax.set_xticks(np.arange(df_plot.shape[1]))
    ax.set_yticks(np.arange(df_plot.shape[0]))

    ax.set_xticklabels(df_plot.columns, fontsize=12, fontweight="bold", color="white", rotation=0)
    ax.set_yticklabels(
        [fr"$\lambda$ = {float(idx):.2f}" for idx in df_plot.index],
        fontsize=11, fontweight="bold", color="white"
    )

    ax.xaxis.tick_top()
    ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, colors="white")

    for i in range(df_plot.shape[0]):
        for j in range(df_plot.shape[1]):
            value = df_plot.iloc[i, j]
            text_color = "white" if value >= 0.25 else "black"
            ax.text(
                j, i, f"{value:.2%}", ha="center", va="center",
                color=text_color, fontsize=10, fontweight="bold" if value >= 0.25 else "normal"
            )

    ax.set_xticks(np.arange(-0.5, df_plot.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, df_plot.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=0.5, alpha=0.30)
    ax.tick_params(which="minor", bottom=False, left=False)

    for spine in ax.spines.values():
        spine.set_visible(False)

    # Title + subtitle as one block so their spacing is governed by line-height
    # rather than separately tuned suptitle/pad values
    ax.set_title(
        r"Portfolio Weights across the Shrinkage Coefficient $\lambda$" "\n"
        r"$\lambda = 0$: historical covariance   $\rightarrow$   $\lambda = 1$: hybrid forward-looking covariance",
        fontsize=16, fontweight="bold", color="white", pad=20, linespacing=1.8
    )
    ax.set_ylabel(r"Shrinkage coefficient $\lambda$", fontsize=14, fontweight="bold", color="white", labelpad=20)

    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=11, colors="white")
    cbar.set_label("Portfolio weight", color="white", fontsize=12, fontweight="bold")

    fig.text(
        0.02, 0.035,
        r"$\lambda$ measures the intensity of the shrinkage toward the hybrid covariance matrix.",
        color="white", fontsize=12, style="italic"
    )

    plt.tight_layout(rect=[0, 0.07, 1, 1])
    return fig


def plot_diversification_across_lambda(lambda_metrics, figsize=(10, 6)):
    """
    Dual-axis line chart of normalized HHI and the effective number of positions (ENP) across
    the lambda grid.
    """

    color_hhi = "#C44E52"
    color_enp = "#4C72B0"

    fig, ax1 = plt.subplots(figsize=figsize, dpi=150)

    line1 = ax1.plot(
        lambda_metrics.index, lambda_metrics["HHI Normalized"],
        marker="s", markersize=5, linewidth=2, color=color_hhi, label="Normalized HHI"
    )
    ax1.set_ylabel("Normalized HHI", fontsize=12, color=color_hhi, fontweight="bold")
    ax1.set_xticks(np.arange(lambda_metrics.index.min(), lambda_metrics.index.max() + 0.01, 0.1))
    ax1.tick_params(axis="y", colors=color_hhi)

    ax2 = ax1.twinx()
    line2 = ax2.plot(
        lambda_metrics.index, lambda_metrics["Effective Number of Positions"],
        marker="o", markersize=5, linewidth=2, color=color_enp, label="Effective Number of Positions"
    )
    ax2.set_ylabel("Effective Number of Positions", fontsize=12, color=color_enp, fontweight="bold")
    ax2.tick_params(axis="y", colors=color_enp)

    ax1.set_xlabel(r"$\lambda$")
    ax1.set_title("Portfolio Diversification across Lambda Values", fontsize=16, fontweight="bold")
    ax1.grid(alpha=0.3)

    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, fontsize=10, loc="upper center", frameon=True)

    plt.tight_layout()
    return fig


def plot_sharpe_across_lambda(lambda_metrics, figsize=(10, 6)):
    """
    Line chart of the Sharpe ratio as a function of the shrinkage coefficient lambda.
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    ax.plot(lambda_metrics.index, lambda_metrics["Sharpe Ratio"], marker="o")
    ax.set_title("Sharpe Ratio across Lambda Values", fontsize=16, weight="bold")
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    return fig


def plot_risk_contribution_heatmap(risk_contrib_grid, figsize=(20, 9)):
    """
    Heatmap of each asset's risk contribution (%) across the full lambda grid.
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    sns.heatmap(
        risk_contrib_grid.T.astype(float) * 100,
        cmap="RdYlGn_r", annot=True, fmt=".1f", annot_kws={"fontsize": 9},
        linewidths=0.3, linecolor="white",
        cbar_kws={"label": "Risk contribution (%)"}, ax=ax
    )
    ax.set_title(r"Risk Contribution by Asset across the Full $\lambda$ Grid", fontweight="bold", pad=15)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Asset")
    plt.tight_layout()
    return fig


def plot_global_weights_heatmap(global_weights, figsize=(20, 14)):
    """
    Heatmap of portfolio weights across the main in-sample configurations (benchmark, FF6,
    hybrid variants).
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    sns.heatmap(
        global_weights, cmap="Blues", annot=True, fmt=".2f",
        linewidths=0.5, linecolor="white",
        cbar_kws={"label": "Portfolio Weight"}, ax=ax
    )
    ax.set_title("Portfolio Weights - Main In-Sample Configurations", fontsize=18, weight="bold", pad=15)
    ax.set_xlabel("")
    ax.set_ylabel("Asset")
    plt.tight_layout()
    return fig


def plot_backtest_weights_stacked(bt_weights, tickers, figsize=(13, 10)):
    """
    Stacked area charts of monthly portfolio weights across a rolling backtest, one subplot per
    configuration in bt_weights.
    """

    fig, axes = plt.subplots(2, 1, figsize=figsize, dpi=150, sharex=True)
    colors = plt.cm.tab20(np.linspace(0, 1, len(tickers)))

    for ax, (label, weights_df) in zip(axes, bt_weights.items()):
        ax.stackplot(
            weights_df.index.astype(str),
            [weights_df[ticker] for ticker in tickers],
            labels=tickers, colors=colors
        )
        ax.set_title(label, fontweight="bold", loc="left", pad=10)
        ax.set_ylabel("Weight")
        ax.set_ylim(0, 1)
        hide_spines(ax)

    step = max(len(weights_df) // 15, 1)
    axes[-1].set_xticks(np.arange(len(weights_df))[::step])
    axes[-1].set_xticklabels(weights_df.index.astype(str)[::step], rotation=45, ha="right")

    axes[0].legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8, frameon=False)
    fig.suptitle("Monthly Weight Allocation across the Backtest", fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 0.88, 0.96])
    return fig


def plot_underwater(drawdowns, figsize=(12, 6)):
    """
    Underwater plot (drawdown from the running maximum) for one or more return series.
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for color, column in zip(colors, drawdowns.columns):
        ax.plot(drawdowns.index.astype(str), drawdowns[column], linewidth=1.5, color=color, label=column)
        ax.fill_between(drawdowns.index.astype(str), drawdowns[column], 0, color=color, alpha=0.25)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Underwater Plot — Drawdown from Running Maximum", fontweight="bold", pad=15)
    ax.set_ylabel("Drawdown (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    hide_spines(ax)

    step = max(len(drawdowns) // 15, 1)
    ax.set_xticks(np.arange(len(drawdowns))[::step])
    ax.set_xticklabels(drawdowns.index.astype(str)[::step], rotation=45, ha="right")

    plt.tight_layout()
    return fig


def plot_ledoit_wolf_bootstrap(lw_result, labels=("Historical Benchmark", "Factor-Based (FF6)"), figsize=(14, 6)):
    """
    Two-panel figure for the Ledoit-Wolf Sharpe-ratio test: the bootstrap distribution of the
    Sharpe gap with its p-value, and each portfolio's own Sharpe ratio with its 95% bootstrap
    confidence interval.
    """

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=150)

    ax1 = axes[0]
    ci_low, ci_high = np.percentile(lw_result["theta_boot"], [2.5, 97.5])

    ax1.hist(lw_result["theta_boot"], bins=40, color="#D3D3D3", edgecolor="white")
    ax1.axvline(lw_result["theta_hat"], color="red", linewidth=2, label=r"$\hat{\theta}$ (observed)")
    ax1.axvline(ci_low, color="black", linestyle="--", linewidth=1.2, label="95% CI")
    ax1.axvline(ci_high, color="black", linestyle="--", linewidth=1.2)

    ax1.set_title(r"Bootstrap distribution of $SR_{bench} - SR_{FF6}$", fontweight="bold")
    ax1.set_xlabel(r"$\theta_b$")
    ax1.set_ylabel("Frequency")
    ax1.legend(frameon=False, loc="upper left")
    ax1.text(
        0.97, 0.95, f"p-value = {lw_result['p_value']:.4f}",
        transform=ax1.transAxes, ha="right", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#AAAAAA")
    )
    hide_spines(ax1)

    ax2 = axes[1]
    sr_values = [lw_result["SR_bench"], lw_result["SR_ff6"]]
    boot_arrays = [lw_result["SR_bench_boot"], lw_result["SR_ff6_boot"]]
    x_pos = np.arange(len(labels))

    for x, sr, boot in zip(x_pos, sr_values, boot_arrays):
        lo, hi = np.percentile(boot, [2.5, 97.5])
        ax2.errorbar(
            x, sr, yerr=[[sr - lo], [hi - sr]],
            fmt="o", color="#4C72B0", markersize=10, ecolor="#4C72B0", elinewidth=2, capsize=6
        )

    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(labels)
    ax2.set_xlim(-0.5, len(labels) - 0.5)
    ax2.set_ylabel("Annualized Sharpe Ratio")
    ax2.set_title("95% bootstrap confidence intervals", fontweight="bold")
    hide_spines(ax2)

    plt.tight_layout()
    return fig


def plot_stress_sharpe_across_lambda(stress_results, figsize=(12, 6)):
    """
    Line chart of the Sharpe ratio across the lambda grid for each stress scenario, against the
    baseline.
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    for label, df in stress_results.items():
        style = dict(linewidth=2.5, linestyle="--", color="black") if label.startswith("Baseline") \
            else dict(linewidth=1.8)
        ax.plot(df.index, df["Sharpe Ratio"], label=label, **style)

    ax.set_title(r"Sharpe Ratio across $\lambda$ — Stress Scenarios vs Baseline", fontweight="bold", pad=15)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    hide_spines(ax)

    plt.tight_layout()
    return fig


def plot_backtest_turnover(turnover_df, figsize=(12, 6)):
    """
    Line chart of month-over-month portfolio turnover across a rolling backtest.
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    for label in turnover_df.columns:
        ax.plot(turnover_df.index.astype(str), turnover_df[label] * 100, marker="o", markersize=3, linewidth=1.1, label=label)

    ax.set_title("Month-over-Month Turnover", fontweight="bold", pad=15)
    ax.set_ylabel("Turnover (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    tick_mask = turnover_df.index.month == 1
    ax.set_xticks(np.where(tick_mask)[0])
    ax.set_xticklabels(turnover_df.index[tick_mask].astype(str), rotation=45, ha="right")
    ax.legend(frameon=False)

    plt.tight_layout()
    return fig


def plot_backtest_concentration(hhi_over_time_df, enp_over_time_df, figsize=(14, 6)):
    """
    Side-by-side line charts of normalized HHI and the effective number of positions (ENP) over
    time, across a rolling backtest.
    """

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=150)

    for label in hhi_over_time_df.columns:
        axes[0].plot(hhi_over_time_df.index.astype(str), hhi_over_time_df[label], marker="o", markersize=3, linewidth=1.1, label=label)
        axes[1].plot(enp_over_time_df.index.astype(str), enp_over_time_df[label], marker="o", markersize=3, linewidth=1.1, label=label)

    axes[0].set_title("Normalized HHI over time", fontweight="bold", pad=10)
    axes[0].set_ylabel("HHI (normalized)")
    axes[1].set_title("Effective Number of Positions over time", fontweight="bold", pad=10)
    axes[1].set_ylabel("Effective Number of Positions")

    tick_mask = hhi_over_time_df.index.month == 1
    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_xticks(np.where(tick_mask)[0])
        ax.set_xticklabels(hhi_over_time_df.index[tick_mask].astype(str), rotation=45, ha="right")

    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)

    plt.tight_layout()
    return fig


def plot_factor_stability(factor_stability_df, figsize=(14, 6)):
    """
    Side-by-side line charts of the average market beta and the average adjusted R² of the FF6
    model, re-estimated over rolling windows.
    """

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=150)

    axes[0].plot(factor_stability_df.index.astype(str), factor_stability_df["Avg Mkt-RF Beta"], marker="o", markersize=3, linewidth=1.1)
    axes[0].set_title("Average Mkt-RF Beta across the Universe", fontweight="bold", pad=10)
    axes[0].set_ylabel(r"Average $\beta_{Mkt-RF}$")

    axes[1].plot(factor_stability_df.index.astype(str), factor_stability_df["Avg R2 (adj)"], marker="o", markersize=3, linewidth=1.1, color="#C44E52")
    axes[1].set_title("Average Adjusted $R^2$ across the Universe", fontweight="bold", pad=10)
    axes[1].set_ylabel(r"Average $R^2_{adj}$")

    tick_mask = factor_stability_df.index.month == 1
    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.3)
        ax.set_xticks(np.where(tick_mask)[0])
        ax.set_xticklabels(factor_stability_df.index[tick_mask].astype(str), rotation=45, ha="right")

    plt.tight_layout()
    return fig


def plot_esg_scores_by_asset(esg_summary, figsize=(20, 10)):
    """
    Grouped bar chart of the composite ESG score and its Environmental / Social / Governance
    components, by asset.
    """

    df_plot = esg_summary.rename(columns={
        "ESG Score": "ESG", "Environmental": "E", "Social": "S", "Governance": "G"
    }).sort_values("ESG", ascending=False).reset_index(drop=True)

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["STIXGeneral", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "axes.titlesize": 24,
        "axes.labelsize": 15,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
    })

    x = np.arange(len(df_plot))
    width_average = 0.3
    width_others = 0.15

    fig, ax = plt.subplots(figsize=figsize)

    bars_esg = ax.bar(x - 1.5 * width_average, df_plot["ESG"], width_average, label="Average ESG", color="#1f77b4")
    bars_e = ax.bar(x - 0.5 * width_others, df_plot["E"], width_others, label="Environmental", alpha=0.5, color="#2ca02c")
    bars_s = ax.bar(x + 0.5 * width_others, df_plot["S"], width_others, label="Social", alpha=0.5, color="#ff7f0e")
    bars_g = ax.bar(x + 1.5 * width_others, df_plot["G"], width_others, label="Governance", alpha=0.5, color="#d62728")

    ax.set_title("ESG Scores by Asset", fontweight="bold", pad=60)
    ax.set_ylabel("Rating")
    ax.set_xlabel("Assets")
    ax.set_ylim(0, 5.2)
    ax.set_xticks(x)
    ax.set_xticklabels(df_plot["Ticker"], rotation=45, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)

    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=4, frameon=False)

    for bars in [bars_esg, bars_e, bars_s, bars_g]:
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9, rotation=90)

    hide_spines(ax)
    plt.tight_layout()
    return fig


def plot_cumulative_performance(growth_df, title, figsize=(12, 6), log_scale=True):
    """
    Cumulative performance (growth of $1) for a set of assets or portfolios, optionally on a log
    scale.
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    for column in growth_df.columns:
        series = growth_df[column].dropna()
        ax.plot(series.index, series.values, linewidth=2, label=column)

    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_ylabel("Growth of $1 (log scale)" if log_scale else "Growth of $1")
    if log_scale:
        ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    hide_spines(ax)

    plt.tight_layout()
    return fig


def plot_underwater_by_asset(drawdowns, title, figsize=(12, 6)):
    """
    Underwater plot (drawdown from the running maximum) for a set of assets, indexed by raw
    dates.
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for color, column in zip(colors, drawdowns.columns):
        series = drawdowns[column].dropna()
        ax.plot(series.index, series.values, linewidth=1.5, color=color, label=column)
        ax.fill_between(series.index, series.values, 0, color=color, alpha=0.25)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontweight="bold", pad=15)
    ax.set_ylabel("Drawdown (%)")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    hide_spines(ax)

    plt.tight_layout()
    return fig


def plot_rolling_beta(rolling_beta, window, figsize=(12, 6)):
    """
    Line chart of rolling beta to the S&P 500 for a set of diversification-candidate assets.
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=150)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for color, column in zip(colors, rolling_beta.columns):
        series = rolling_beta[column].dropna()
        ax.plot(series.index.astype(str), series.values, linewidth=1.5, color=color, label=column)

    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Assets for extra diversification test — Rolling {window}-Month Beta to S&P 500", fontweight="bold", pad=15)
    ax.set_ylabel(r"Rolling $\beta$ to SPY")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)

    tick_mask = rolling_beta.dropna(how="all").index.month == 1
    tick_index = rolling_beta.dropna(how="all").index
    ax.set_xticks(np.where(tick_mask)[0])
    ax.set_xticklabels(tick_index[tick_mask].astype(str), rotation=45, ha="right")

    hide_spines(ax)
    plt.tight_layout()
    return fig


def plot_backtest_cumulative_performance(cum_perf, figsize=(12, 6)):
    """
    Rolling-backtest cumulative performance curves, with a marker for the start of the AI-driven
    bull cycle (Jan. 2023).
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    for column in cum_perf.columns:
        ax.plot(cum_perf.index.astype(str), cum_perf[column], linewidth=2, label=column)

    ax.set_title("Rolling Backtest — Cumulative Performance", fontweight="bold", pad=15)
    ax.set_ylabel("Growth of $1")
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    ai_cycle_pos = cum_perf.index.get_indexer([pd.Period("2023-01", freq="M")])[0]
    ax.axvline(ai_cycle_pos - 0.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.8)
    ax.text(
        ai_cycle_pos - 0.5, ax.get_ylim()[1], "  AI cycle start (Jan. 2023)",
        rotation=90, va="top", ha="left", fontsize=9, color="gray"
    )

    ax.legend(frameon=False)
    hide_spines(ax)

    step = max(len(cum_perf) // 15, 1)
    ax.set_xticks(np.arange(len(cum_perf))[::step])
    ax.set_xticklabels(cum_perf.index.astype(str)[::step], rotation=45, ha="right")

    plt.tight_layout()
    return fig


def plot_backtest_rolling_sharpe(rolling_sharpe, figsize=(12, 6)):
    """
    Line chart of the rolling 12-month Sharpe ratio across backtest configurations.
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    for column in rolling_sharpe.columns:
        ax.plot(rolling_sharpe.index.astype(str), rolling_sharpe[column], linewidth=2, label=column)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title("Rolling 12-Month Sharpe Ratio", fontweight="bold", pad=15)
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    hide_spines(ax)

    step = max(len(rolling_sharpe) // 15, 1)
    ax.set_xticks(np.arange(len(rolling_sharpe))[::step])
    ax.set_xticklabels(rolling_sharpe.index.astype(str)[::step], rotation=45, ha="right")

    plt.tight_layout()
    return fig


def plot_esg_sharpe_vs_constraint(esg_thresholds, sharpe_values, figsize=(10, 6)):
    """
    Line chart of the rolling-backtest Sharpe ratio against increasingly strict ESG minimum-
    score constraints.
    """

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    x_labels = ["FF6"] + [f"≥ {thr}" for thr in esg_thresholds]
    ax.plot(x_labels, sharpe_values, marker="o", markersize=6, linewidth=2, color="#1f77b4")

    ax.set_title("Rolling Backtest — Sharpe Ratio vs. ESG Constraint", fontweight="bold", pad=15)
    ax.set_xlabel("ESG minimum-score constraint")
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    hide_spines(ax)

    plt.tight_layout()
    return fig


def plot_lambda_elbow_analysis(sharpe_retained, enp_retained, hhi_retained, gap_enp, gap_hhi,
                                elbow_enp, elbow_hhi, selected_lambda=0.25, figsize=(14, 6)):
    """
    Two-panel elbow analysis of the Sharpe/diversification trade-off across the lambda grid: the
    trade-off curve itself, and each metric's gap to the proportional-degradation diagonal, with
    the selected lambda and each curve's elbow point marked.
    """

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=150)

    ax1 = axes[0]
    ax1.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.2, label="Proportional degradation")
    ax1.plot(sharpe_retained, enp_retained, marker="o", markersize=4, color="#4C72B0", label="Sharpe vs ENP")
    ax1.plot(sharpe_retained, hhi_retained, marker="o", markersize=4, color="#C44E52", label="Sharpe vs HHI")
    ax1.scatter([sharpe_retained.loc[selected_lambda]], [enp_retained.loc[selected_lambda]], color="black", s=90, zorder=5, marker="D", label=rf"$\lambda={selected_lambda}$ (selected)")
    ax1.scatter([sharpe_retained.loc[elbow_enp]], [enp_retained.loc[elbow_enp]], color="#4C72B0", s=90, zorder=5, edgecolor="black")
    ax1.scatter([sharpe_retained.loc[elbow_hhi]], [hhi_retained.loc[elbow_hhi]], color="#C44E52", s=90, zorder=5, edgecolor="black")
    ax1.set_xlabel("Retained Sharpe Ratio (normalized)")
    ax1.set_ylabel("Retained Diversification (normalized)")
    ax1.set_title("Sharpe / Diversification Trade-off Curve", fontweight="bold")
    ax1.legend(frameon=False, fontsize=9, loc="lower left")
    ax1.invert_xaxis()
    hide_spines(ax1)

    ax2 = axes[1]
    ax2.plot(gap_enp.index, gap_enp.values, marker="o", markersize=4, color="#4C72B0", label="Gap (ENP)")
    ax2.plot(gap_hhi.index, gap_hhi.values, marker="o", markersize=4, color="#C44E52", label="Gap (HHI)")
    ax2.axvline(selected_lambda, color="black", linestyle="--", linewidth=1.2, label=rf"$\lambda={selected_lambda}$ (selected)")
    ax2.axvline(elbow_enp, color="#4C72B0", linestyle=":", linewidth=1.2)
    ax2.axvline(elbow_hhi, color="#C44E52", linestyle=":", linewidth=1.2)
    ax2.set_xlabel(r"$\lambda$")
    ax2.set_ylabel("Gap relative to the diagonal")
    ax2.set_title("Elbow Location", fontweight="bold")
    ax2.legend(frameon=False, fontsize=9)
    hide_spines(ax2)

    plt.tight_layout()
    return fig


def plot_sharpe_lambda_snapshot_comparison(lambda_metrics_current, archive_path,
                                            current_label, archive_label,
                                            figsize=(10, 6)):
    """
    Sharpe-vs-lambda curve for the current implied-vol snapshot compared against an archived
    one, to illustrate how sensitive the curve's shape is to the snapshot date.
    """

    with open(archive_path) as f:
        archive = json.load(f)
    archive_df = pd.DataFrame(archive["lambda_metrics"]).set_index("Lambda")

    fig, ax = plt.subplots(figsize=figsize, dpi=150)

    ax.plot(archive_df.index, archive_df["Sharpe Ratio"], linewidth=2, marker="o",
            markersize=4, label=archive_label, color="#999999", linestyle="--")
    ax.plot(lambda_metrics_current.index, lambda_metrics_current["Sharpe Ratio"], linewidth=2,
            marker="o", markersize=4, label=current_label, color="#4C72B0")

    ax.set_title("Sharpe Ratio Across the Shrinkage Grid — Two Implied Vol Snapshots", fontweight="bold", pad=15)
    ax.set_xlabel(r"$\lambda$")
    ax.set_ylabel("Sharpe Ratio")
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    ax.legend(frameon=False)
    hide_spines(ax)

    plt.tight_layout()
    return fig


def plot_risk_free_rate(rf_annualized, figsize=(8, 4)):
    """
    Line chart of the annualized risk-free rate over time (from monthly-sampled Fama-French
    data), with its sample mean marked.
    """

    import matplotlib.dates as mdates

    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    x = rf_annualized.index
    if hasattr(x, "to_timestamp"):
        x = x.to_timestamp()

    ax.plot(x, rf_annualized.values * 100, marker="+", markersize=8, linewidth=1.5, color="#4C72B0")

    mean_value = rf_annualized.mean() * 100
    ax.axhline(mean_value, linestyle="--", linewidth=2, color="#4C72B0", label=f"Mean: {mean_value:.2f}%")

    ax.xaxis.set_major_locator(mdates.YearLocator(2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax.set_title("Fama-French Risk-Free Rate (Annualized)", fontsize=18, weight="bold", pad=15)
    ax.set_xlabel("Date")
    ax.set_ylabel("Risk-Free Rate (%)")
    ax.legend(frameon=False, fontsize=11, loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    hide_spines(ax)

    plt.tight_layout()
    return fig, ax
