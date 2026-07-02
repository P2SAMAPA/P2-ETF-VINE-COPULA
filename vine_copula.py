import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import config

def vine_copula_score(returns, macro_df):
    """
    Fit R‑vine copula to ETF returns and compute conditional quantile for each ETF.
    Multiply by momentum factor (1 + last_return) to enhance return potential.
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
    # Convert to pseudo-observations (ranks)
    from scipy.stats import rankdata
    u = np.apply_along_axis(rankdata, 0, returns) / (len(returns) + 1)
    # Estimate macro factor (composite) using PCA
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    macro_scaled = scaler.fit_transform(macro_df)
    pca = PCA(n_components=1)
    macro_factor = pca.fit_transform(macro_scaled).flatten()
    macro_factor = (macro_factor - macro_factor.min()) / (macro_factor.max() - macro_factor.min() + 1e-8)
    # Fit vine copula
    try:
        import pyvinecopulib as pv
        # Select families
        families = []
        for fam in config.COPULA_FAMILIES:
            if fam == "gaussian":
                families.append(pv.BicopFamily.gaussian)
            elif fam == "clayton":
                families.append(pv.BicopFamily.clayton)
            elif fam == "gumbel":
                families.append(pv.BicopFamily.gumbel)
            elif fam == "joe":
                families.append(pv.BicopFamily.joe)
            elif fam == "frank":
                families.append(pv.BicopFamily.frank)
        # Fit vine with automatic family selection - pass arguments to constructor
        d = u.shape[1]
        vine = pv.Vinecop(d, family_set=families, selection_criterion='aic')
        vine.fit(u)
        # Compute conditional quantile for each ETF given macro state
        scores = {}
        tickers = returns.columns
        n = len(tickers)
        pair_strength = np.zeros(n)
        # Get the vine structure and pair copulas
        for i in range(n):
            tau_sum = 0.0
            for tree in vine.trees:
                for edge in tree:
                    if edge.get_var_names() is not None:
                        var_names = edge.get_var_names()
                        if tickers[i] in var_names:
                            tau_sum += abs(edge.copula.tau())
            pair_strength[i] = tau_sum
        # Normalise pair_strength
        if pair_strength.max() > 0:
            pair_strength = pair_strength / pair_strength.max()
        # Get last returns for momentum
        last_returns = returns.iloc[-1].values
        # Combine: score = vine_score × (1 + momentum)
        for i, ticker in enumerate(tickers):
            vine_score = pair_strength[i] * (0.5 + macro_factor[-1] * 0.5)
            # Momentum factor: 1 + last_return (clipped to [0.5, 2.0] to avoid extremes)
            momentum = 1.0 + last_returns[i]
            momentum = max(0.5, min(2.0, momentum))
            scores[ticker] = float(vine_score * momentum)
        return scores
    except Exception as e:
        print(f"Vine fitting failed: {e}")
        # Fallback: use pair correlations as scores
        corr = returns.corr().abs().mean(axis=1)
        last_returns = returns.iloc[-1].values
        scores = {}
        for i, ticker in enumerate(returns.columns):
            momentum = 1.0 + last_returns[i]
            momentum = max(0.5, min(2.0, momentum))
            scores[ticker] = float(corr.iloc[i] * momentum)
        return scores
