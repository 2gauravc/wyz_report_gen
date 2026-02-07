import pandas as pd 
import numpy as np

def assign_scores(s: pd.Series, bins_count: int = 5) -> pd.Series:
    # Assign scores 1..bins_count based on equal-width bins within group
    vals = s.copy()
    numeric = pd.to_numeric(vals, errors="coerce")
    if numeric.dropna().empty:
        return pd.Series([np.nan] * len(s), index=s.index)
    mn = numeric.min()
    mx = numeric.max()
    if mn == mx:
        return pd.Series([bins_count if not np.isnan(v) else np.nan for v in numeric], index=s.index)
    bins = np.linspace(mn, mx, bins_count + 1)
    labels = list(range(1, bins_count + 1))
    scored = pd.cut(numeric, bins=bins, labels=labels, include_lowest=True)
    return scored.astype(float)
