"""
Configuration for P2-ETF-T-COPULA engine.
"""

import os
from datetime import datetime

# --- Hugging Face Repositories ---
HF_DATA_REPO = "P2SAMAPA/fi-etf-macro-signal-master-data"
HF_DATA_FILE = "master_data.parquet"
HF_OUTPUT_REPO = "P2SAMAPA/p2-etf-t-copula-results"

# --- Universe Definitions ---
FI_COMMODITIES_TICKERS = ["TLT", "VCIT", "LQD", "HYG", "VNQ", "GLD", "SLV"]
EQUITY_SECTORS_TICKERS = [
    "SPY", "QQQ", "XLK", "XLF", "XLE", "XLV",
    "XLI", "XLY", "XLP", "XLU", "GDX", "XME",
    "IWF", "XSD", "XBI", "IWM"
]
ALL_TICKERS = list(set(FI_COMMODITIES_TICKERS + EQUITY_SECTORS_TICKERS))

UNIVERSES = {
    "FI_COMMODITIES": FI_COMMODITIES_TICKERS,
    "EQUITY_SECTORS": EQUITY_SECTORS_TICKERS,
    "COMBINED": ALL_TICKERS
}

# --- Macro Columns ---
MACRO_COLS = ["VIX", "DXY", "T10Y2Y", "TBILL_3M"]

# --- Copula Parameters ---
DAILY_LOOKBACK = 504               # Days for daily training
GLOBAL_TRAIN_START = "2008-01-01"  # Global training start

# --- Simulation ---
N_SIMULATIONS = 100000             # Increased for more precise risk metrics
TAIL_ADJUSTMENT_LAMBDA = 1.0       # Weight for ES95 penalty
RISK_FREE_RATE_ANNUAL = 0.02

# --- Parametric Marginals ---
USE_GARCH = True                   # Use GARCH(1,1) + skew‑t instead of empirical CDF
GARCH_P = 1
GARCH_Q = 1
GARCH_DIST = "skewt"              # skew‑t distribution for GARCH residuals

# --- Bootstrap ---
BOOTSTRAP_SAMPLES = 1000           # Bootstrap resamples for VaR/ES confidence intervals

# --- Conditional Expected Return ---
MOMENTUM_WINDOW = 21               # Days for forward‑looking return signal
MIN_OBSERVATIONS = 252             # Minimum data required
GLOBAL_MIN_OBSERVATIONS = 1008     # Minimum for global training

# --- Shrinking Windows ---
SHRINKING_WINDOW_START_YEARS = list(range(2008, 2025))

# --- Date Handling ---
TODAY = datetime.now().strftime("%Y-%m-%d")

# --- Optional: Hugging Face Token ---
HF_TOKEN = os.environ.get("HF_TOKEN", None)
