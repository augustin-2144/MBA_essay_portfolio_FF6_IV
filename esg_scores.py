import pandas as pd


CORE_COLUMNS = ["Ticker", "Company", "ESG Score", "Environmental", "Social", "Governance"]


def load_lseg_esg_table(path, tickers=None, sheet_name="ESG Summary"):
    """
    Load the LSEG ESG table exported from the project Excel file.

    The scores are expected to be on LSEG's 0-5 scale.
    If tickers is provided, the table is validated and reindexed to match
    the portfolio universe exactly.
    """
    df = pd.read_excel(path, sheet_name=sheet_name)

    if "Ticker" not in df.columns:
        raise ValueError("The ESG file must contain a 'Ticker' column.")

    df["Ticker"] = df["Ticker"].astype(str).str.upper().str.strip()
    df = df.set_index("Ticker", drop=False)

    missing_columns = [col for col in CORE_COLUMNS if col not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing ESG columns: {missing_columns}")

    numeric_columns = [col for col in df.columns if col not in ["Ticker", "Company", "Source"]]
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    if tickers is not None:
        tickers = [ticker.upper().strip() for ticker in tickers]
        missing_tickers = [ticker for ticker in tickers if ticker not in df.index]
        if missing_tickers:
            raise ValueError(f"Missing ESG scores for tickers: {missing_tickers}")
        df = df.reindex(tickers)

    if df[CORE_COLUMNS[2:]].isna().any().any():
        bad_rows = df[df[CORE_COLUMNS[2:]].isna().any(axis=1)].index.tolist()
        raise ValueError(f"Missing ESG values for tickers: {bad_rows}")

    return df


def load_lseg_esg_scores(path, tickers, score_column="ESG Score", sheet_name="ESG Summary"):
    """
    Return a pd.Series of LSEG ESG scores aligned with the portfolio universe.
    """
    table = load_lseg_esg_table(path=path, tickers=tickers, sheet_name=sheet_name)

    if score_column not in table.columns:
        raise ValueError(f"Column '{score_column}' not found in ESG table.")

    scores = table[score_column].astype(float)
    scores.name = score_column
    return scores


def build_esg_summary_for_display(path, tickers, sheet_name="ESG Summary"):
    """
    Return the columns useful for notebook display and reporting.
    """
    table = load_lseg_esg_table(path=path, tickers=tickers, sheet_name=sheet_name)
    display_columns = [col for col in CORE_COLUMNS if col in table.columns]
    return table[display_columns].copy()
