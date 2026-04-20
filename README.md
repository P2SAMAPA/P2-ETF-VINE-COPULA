# P2-ETF-VINE-COPULA

**Vine Copula for High‑Dimensional Tail Dependence & Risk‑Adjusted ETF Ranking**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-VINE-COPULA/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-VINE-COPULA/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--vine--copula--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-vine-copula-results)

## Overview

`P2-ETF-VINE-COPULA` models the joint distribution of 23 ETFs using **regular‑vine copulas**, capturing complex asymmetric tail dependencies beyond standard parametric copulas. From Monte Carlo simulations, it computes tail‑adjusted expected returns and ranks ETFs for next‑day trading.

## Methodology

1. **Marginal Modeling**: Empirical CDF or skew‑t for each ETF's returns.
2. **Vine Copula Fit**: R‑vine structure selection and pair‑copula estimation.
3. **Joint Simulation**: 10,000 scenarios of next‑day returns.
4. **Risk‑Adjusted Score**: `Score = E[Return] - λ * |ES_95|`.

## Universe
FI/Commodities, Equity Sectors, Combined (23 ETFs)

## Usage
```bash
pip install -r requirements.txt
python trainer.py
streamlit run streamlit_app.py
