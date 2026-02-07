import pandas as pd
import numpy as np
import re

def display_value(v, na="NA"):
    if v is None:
        return na
    if isinstance(v, float) and np.isnan(v):
        return na
    if pd.isna(v):
        return na
    return v

def sanitize_fname(s: str) -> str:
        return re.sub(r"[^0-9A-Za-z._-]", "_", s)[:80]
