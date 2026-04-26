# P2-ETF-T-COPULA

**Multivariate Student‑t Copula – Tail‑Dependence‑Aware ETF Ranking**

[![Daily Run](https://github.com/P2SAMAPA/P2-ETF-T-COPULA/actions/workflows/daily_run.yml/badge.svg)](https://github.com/P2SAMAPA/P2-ETF-T-COPULA/actions/workflows/daily_run.yml)
[![Hugging Face Dataset](https://img.shields.io/badge/🤗%20Dataset-p2--etf--t--copula--results-blue)](https://huggingface.co/datasets/P2SAMAPA/p2-etf-t-copula-results)

## Overview

`P2-ETF-T-COPULA` fits an `n`‑dimensional **Student‑t copula** to ETF returns (Demarta & McNeil algorithm). The multivar‑iate t‑copula explicitly captures **tail dependence** — the tendency of assets to crash together — which a Gaussian copula cannot. Monte Carlo simulation produces full return scenarios, and the engine ranks ETFs using a **t‑copula‑adjusted score**:

`Score = Momentum (21‑day annualised) − λ × |ES₉₅|`

Three views are produced per universe:

- **Daily (504d):** Trained on the most recent 2 years to capture the current regime.
- **Global (2008‑YTD):** Trained on the entire available history for long‑term tail estimation.
- **Shrinking Windows Consensus:** The most frequently selected ETF across 17 rolling windows (2008‑2024).

## Methodology

1. **Marginal modelling:** Empirical CDF per ETF (no parametric assumption).
2. **Copula:** Multivariate Student‑t; degrees of freedom estimated from tail dependence coefficients.
3. **Simulation:** Cholesky decomposition + χ²/ν scaling → correlated t‑variates → mapped to uniforms → inverted via empirical quantiles.
4. **Scoring:** Conditional momentum minus tail penalty.

## Why t‑Copula Instead of Vine Copula

The original VINE‑COPULA engine depended on `vinecopulib`, a C++ library that cannot be installed on GitHub Actions. The t‑copula is implemented in pure Python (SciPy/NumPy), installs natively, and captures the essential tail‑dependence feature that was missing from the Gaussian fallback.

## Universe
FI/Commodities, Equity Sectors, Combined (23 ETFs)
