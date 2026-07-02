# Vine Copula Decomposition for ETFs

Fits an R‑vine copula to ETF returns, selecting the optimal bivariate copula family (Gaussian, Clayton, Gumbel, Joe, Frank) for each pair via AIC. The per‑ETF score is the conditional quantile given the macro state – a measure of asymmetric tail dependence.

## Features
- Three ETF universes (FI/Commodities, Equity Sectors, Combined)
- Seven rolling windows (63–4536 days)
- R‑vine copula with AIC‑selected bivariate families
- Conditional quantile given composite macro factor
- Score = conditional dependence strength
- Two‑tab Streamlit dashboard (auto best, manual)
- Results stored on Hugging Face: `P2SAMAPA/p2-etf-vine-copula-results`

## Usage

1. Set `HF_TOKEN` environment variable.
2. Install dependencies: `pip install -r requirements.txt` (requires `pyvinecopulib`)
3. Run training: `python train.py` (slower due to vine fitting)
4. Launch dashboard: `streamlit run streamlit_app.py`

## Interpretation

- High score → ETF has strong asymmetric tail dependence with the universe under current macro.
- Low score → ETF is independent or has weak tail dependence.

## Requirements

See `requirements.txt`.
