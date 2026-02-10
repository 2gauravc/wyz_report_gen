import re 

def _clean_col(col: str) -> str:
    return re.sub(r"[^a-z0-9]", "", col.lower())

def find_column(columns, patterns):
    cleaned_cols = {col: _clean_col(col) for col in columns}
    cleaned_pats = [_clean_col(pat) for pat in patterns if pat]  # <-- key fix
    for col, c in cleaned_cols.items():
        for pat in cleaned_pats:
            if pat in c:
                return col
    return None
