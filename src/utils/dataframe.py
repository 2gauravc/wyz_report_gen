import pandas as pd 
import numpy as np

def safe_numeric_col(
    source_df: pd.DataFrame,
    col_name: str,
    default=np.nan
) -> pd.Series:
    """
    Safely extract a column and convert it to numeric.
    Returns a Series aligned to source_df.index.
    """
    if col_name in source_df.columns:
        return pd.to_numeric(source_df[col_name], errors="coerce")
    else:
        return pd.Series(default, index=source_df.index)
