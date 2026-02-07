import re 

def _clean_col(col: str) -> str:
    return re.sub(r"[^a-z0-9]", "", col.lower())

def find_column(columns, patterns):
    cleaned = {col: _clean_col(col) for col in columns}
    for col, c in cleaned.items():
        for pat in patterns:
            if pat in c:
                return col
    return None
