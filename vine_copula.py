import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def vine_copula_score(returns, macro_df):
    """
    Simplified vine copula using Gaussian copula.
    Computes conditional quantile via correlation matrix and macro factor.
    """
    if len(returns) < 20 or macro_df is None or len(macro_df) < 20:
        return {ticker: 0.0 for ticker in returns.columns}
    # Align lengths
    min_len = min(len(returns), len(macro_df))
    returns = returns[:min_len]
    macro_df = macro_df.iloc[:min_len]
    # Remove NaN
    mask = ~(np.isnan(returns).any(axis=1) | np.isnan(macro_df).any(axis=1))
    returns = returns[mask]
    macro_df = macro_df[mask]
    if len(returns) < 20:
        return {ticker: 0.0 for ticker in returns.columns}
    # Estimate macro factor (composite) using PCA
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    macro_factor = pca.fit_transform(macro_scaled).flatten()
    macro_factor = (macro_factor - macro_factor.min()) / (macro_factor.max() - macro_factor.min() + 1e-8)
    # Compute correlation matrix (Gaussian copula)
    corr = returns.corr().values
    # For each ETF, compute the conditional quantile given macro
    tickers = returns.columns
    n = len(tickers)
    # Use the macro factor as a conditioning variable
    # We'll compute the partial correlation between each ETF and macro
    # Then use it to compute the conditional quantile
    scores = {}
    # Standardise returns
    ret_scaled = (returns - returns.mean()) / returns.std()
    # For each ETF, compute its correlation with macro factor
    macro_factor_series = pd.Series(macro_factor, index=returns.index)
    scores_raw = {}
    for i, ticker in enumerate(tickers):
        # Correlation between ETF and macro factor
        corr_etf_macro = ret_scaled[ticker].corr(macro_factor_series)
        if np.isnan(corr_etf_macro):
            corr_etf_macro = 0.0
        # Conditional quantile: E[ETF | macro] = mean + corr * (macro - mean_macro) / std_macro * std_etf
        # Simplified: use the correlation as the score
        scores_raw[ticker] = abs(corr_etf_macro)
    # Normalise scores
    max_score = max(scores_raw.values()) if scores_raw else 1.0
    if max_score > 0:
        scores_raw = {k: v / max_score for k, v in scores_raw.items()}
    # Get last returns for momentum
    last_returns = returns.iloc[-1].values
    # Combine: score = vine_score × (1 + momentum)
    scores = {}
    for i, ticker in enumerate(tickers):
        vine_score = scores_raw[ticker] * (0.5 + macro_factor[-1] * 0.5)
        momentum = 1.0 + last_returns[i]
        momentum = max(0.5, min(2.0, momentum))
        scores[ticker] = float(vine_score * momentum)
    return scores
