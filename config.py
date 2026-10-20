from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent
ESG_DATA_PATH = BASE_DIR / "LSEG_ESG_scores.xlsx"
IMPLIED_VOLS_SNAPSHOT_PATH = BASE_DIR / "implied_vols_snapshot.csv"
PRICES_SNAPSHOT_PATH = BASE_DIR / "prices_snapshot.csv"
SPY_PRICES_SNAPSHOT_PATH = BASE_DIR / "spy_prices_snapshot.csv"
FF6_FACTORS_SNAPSHOT_PATH = BASE_DIR / "ff6_factors_snapshot.csv"
SPY_IVV_SNAPSHOT_PATH = BASE_DIR / "spy_ivv_snapshot.csv"
TSLA_DAILY_PRICES_SNAPSHOT_PATH = BASE_DIR / "tsla_daily_prices_snapshot.csv"
TSLA_PE_SNAPSHOT_PATH = BASE_DIR / "tsla_pe_snapshot.csv"
CRYPTO_PRICES_SNAPSHOT_PATH = BASE_DIR / "crypto_prices_snapshot.csv"

# Study period
START_DATE = "2015-01-01"
END_DATE = "2026-06-30" 

# Portfolio universe
TICKERS = [
    # Technology / AI
    "AAPL",
    "MSFT",
    "NVDA",
    "GOOGL",
    "META",
    "AMZN",

    # Semiconductors / AI infrastructure
    "AVGO",

    # Financials
    "JPM",
    "GS",
    "BAC",

    # Consumer Staples
    "KO",
    "PEP",
    "PG",

    # Healthcare
    "JNJ",
    "UNH",

    # Energy
    "XOM",
    "CVX",

    # Industrials
    "CAT",
    "HON",
]

COMPANY_NAMES = {
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corp",
    "NVDA": "NVIDIA Corp",
    "GOOGL": "Alphabet Inc",
    "META": "Meta Platforms Inc",
    "AMZN": "Amazon.com Inc",
    "AVGO": "Broadcom Inc",
    "JPM": "JPMorgan Chase & Co",
    "GS": "Goldman Sachs Group Inc",
    "BAC": "Bank of America Corp",
    "KO": "Coca-Cola Co",
    "PEP": "PepsiCo Inc",
    "PG": "Procter & Gamble Co",
    "JNJ": "Johnson & Johnson",
    "UNH": "UnitedHealth Group Inc",
    "XOM": "Exxon Mobil Corp",
    "CVX": "Chevron Corp",
    "CAT": "Caterpillar Inc",
    "HON": "Honeywell International Inc",
}

SECTOR_MAP = {
    "AAPL": "Technology",
    "MSFT": "Technology",
    "NVDA": "Technology",
    "GOOGL": "Technology",
    "META": "Technology",
    "AMZN": "Consumer Discretionary",
    "AVGO": "Semiconductors",
    "JPM": "Financials",
    "GS": "Financials",
    "BAC": "Financials",
    "KO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "PG": "Consumer Staples",
    "JNJ": "Healthcare",
    "UNH": "Healthcare",
    "XOM": "Energy",
    "CVX": "Energy",
    "CAT": "Industrials",
    "HON": "Industrials",
}

# Crypto-assets (descriptive illustration only, Chapitre 8)
CRYPTO_TICKERS = ["BTC-USD", "ETH-USD"]

# Diversification candidates: bonds, gold, REITs (descriptive illustration only, Chapitre 8)
DIVERSIFICATION_TICKERS = ["TLT", "GLD", "VNQ"]
DIVERSIFICATION_PRICES_SNAPSHOT_PATH = BASE_DIR / "diversification_prices_snapshot.csv"

# LSEG ESG settings
ESG_SHEET_NAME = "ESG Summary"
ESG_SCORE_COLUMN = "ESG Score"

# Real-money illustration amount (Chapitre 8 - ESG)
AMOUNT_INVESTED = 19_200  # price of the author's MBA, in $

# Implied volatility settings
TARGET_DAYS = 30

# Return settings
RETURN_METHOD = "simple"

# Annualization factor
FREQUENCY = 12