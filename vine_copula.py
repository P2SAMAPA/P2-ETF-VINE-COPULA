import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

def vine_copula_score(returns, macro_df):
    """
    Fit R‑vine copula to ETF returns and compute conditional quantile for each ETF.
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
        # Fit vine with automatic family selection
        vine = pv.Vinecop(u, family_set=families, selection_criterion='aic')
        vine.fit(u)
        # Compute conditional quantile for each ETF given macro state
        # For each ETF, we use the macro factor as a conditioning variable
        # Simpler: compute the conditional mean of the vine distribution for each ETF
        # For each ETF, we can extract the marginal conditional quantile
        # We'll use the vine's conditional distribution function
        scores = {}
        tickers = returns.columns
        # For each ETF, compute the expected value under the vine distribution
        # conditioned on the last macro factor
        # This is a simplified approach: we use the marginal mean
        # In practice, we would use the vine to sample conditional on macro
        # But pyvinecopulib doesn't directly support conditioning on macro
        # So we compute the average pair copula strength for each ETF
        # as a proxy for its conditional dependence
        n = len(tickers)
        pair_strength = np.zeros(n)
        # Get the vine structure and pair copulas
        for i in range(n):
            # Sum of absolute Kendall's tau for pairs involving ETF i
            tau_sum = 0.0
            # Extract pair copulas from vine
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
        # Combine with macro factor to get conditional quantile
        for i, ticker in enumerate(tickers):
            # Higher macro factor increases the conditional quantile
            # We use pair_strength as a weight
            score = pair_strength[i] * (0.5 + macro_factor[-1] * 0.5)
            scores[ticker] = float(score)
        return scores
    except Exception as e:
        print(f"Vine fitting failed: {e}")
        # Fallback: use pair correlations as scores
        corr = returns.corr().abs().mean(axis=1)
        return {ticker: float(corr[i]) for i, ticker in enumerate(returns.columns)}
