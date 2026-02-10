from __future__ import annotations
from typing import Any, Dict, Tuple
import numpy as np
import pandas as pd

from utils.columns import find_column
from utils.dataframe import safe_numeric_col

def _series_string(df: pd.DataFrame, col: str | None) -> pd.Series:
    if col and col in df.columns:
        return df[col].astype("string")
    return pd.Series(pd.NA, index=df.index, dtype="string")

def _apply_string_normalizers(s: pd.Series, norm: dict) -> pd.Series:
    if norm.get("strip"):
        s = s.astype("string").str.strip()
    if norm.get("lower"):
        s = s.astype("string").str.lower()
    if norm.get("title_case"):
        s = s.astype("string").str.title()
    # mapping (for gender etc.)
    mapping = norm.get("map")
    if isinstance(mapping, dict):
        # map expects keys to match current values (often lowercased)
        s = s.map(lambda x: mapping.get(x, x) if pd.notna(x) else x).astype("string")
    return s

def _coerce_dtype(df: pd.DataFrame, src_col: str | None, dtype: str) -> pd.Series:
    dtype = (dtype or "").lower()
    if dtype == "number":
        return safe_numeric_col(df, src_col)
    if dtype in ("string", "category"):
        return _series_string(df, src_col)
    # default fall back
    return _series_string(df, src_col)

def _derive_from_regex(source: pd.Series, regex: str, group: int | None = None, cast: str | None = None) -> pd.Series:
    extracted = source.astype("string").str.extract(regex)
    if extracted is None or extracted.empty:
        return pd.Series(pd.NA, index=source.index)
    if group is None:
        s = extracted.iloc[:, 0]
    else:
        # pandas extract returns columns 0..n-1; group=2 means column index 1
        s = extracted.iloc[:, group - 1] if group >= 1 else extracted.iloc[:, 0]
    if cast == "int":
        return pd.to_numeric(s, errors="coerce").astype("Int64")
    if cast == "float":
        return pd.to_numeric(s, errors="coerce").astype(float)
    return s

def build_work_df_from_config(df: pd.DataFrame, cfg: dict) -> Tuple[pd.DataFrame, dict]:
    """
    Returns:
      work_df: standardized dataframe with entity + measurements + derived_attributes
      col_map: mapping of dataset field -> source column name (or None if derived/missing)
    """
    dataset_cfg = cfg.get("dataset", {})
    entity_cfg = dataset_cfg.get("entity_columns", {}) or {}
    meas_cfg = dataset_cfg.get("measurement_columns", {}) or {}
    derived_cfg = dataset_cfg.get("derived_attributes", {}) or {}

    work = pd.DataFrame(index=df.index)
    col_map: Dict[str, str | None] = {}

    # --------
    # entity_columns
    # --------
    for field, spec in entity_cfg.items():
        patterns = spec.get("col_name_patterns", []) or []
        src = find_column(df.columns, patterns) if patterns else None
        col_map[field] = src

        series = _coerce_dtype(df, src, spec.get("dtype", "string"))

        norm = spec.get("normalize_values") or {}
        if isinstance(norm, dict) and spec.get("dtype", "").lower() in ("string", "category"):
            series = _apply_string_normalizers(series, norm)

        work[field] = series

        # derive if requested (e.g., grade/section from class)
        derive = spec.get("derive")
        if isinstance(derive, dict):
            from_field = derive.get("from")
            method = derive.get("method")
            if from_field and method == "regex" and from_field in work.columns:
                regex = derive.get("regex")
                group = derive.get("group")  # optional
                cast = derive.get("cast")    # optional
                work[field] = _derive_from_regex(work[from_field], regex, group=group, cast=cast)

    # Provide srno default if missing (optional field)
    if "srno" in entity_cfg and (work["srno"].isna().all() or len(work["srno"]) == 0):
        work["srno"] = pd.RangeIndex(start=1, stop=len(df) + 1).astype(str)

    # --------
    # measurement_columns
    # --------
    for field, spec in meas_cfg.items():
        patterns = spec.get("col_name_patterns", []) or []
        src = find_column(df.columns, patterns) if patterns else None
        col_map[field] = src

        series = _coerce_dtype(df, src, spec.get("dtype", "number"))

        norm = spec.get("normalize_values") or {}
        if isinstance(norm, dict) and spec.get("dtype", "").lower() in ("string", "category"):
            series = _apply_string_normalizers(series, norm)

        work[field] = series

    # --------
    # derived_attributes (BMI etc.)
    # --------
    for field, spec in derived_cfg.items():
        depends = spec.get("depends_on", []) or []
        compute = (spec.get("compute") or {})
        method = compute.get("method")
        round_to = compute.get("round")

        if method == "bmi":
            # expects height_cm + weight_kg in dataset
            h = pd.to_numeric(work[depends[0]], errors="coerce") if len(depends) > 0 else np.nan
            w = pd.to_numeric(work[depends[1]], errors="coerce") if len(depends) > 1 else np.nan
            bmi = w / ((h / 100.0) ** 2)
            if round_to is not None:
                bmi = bmi.round(int(round_to))
            work[field] = bmi
            col_map[field] = None
        else:
            raise ValueError(f"Unsupported derived_attributes.compute.method: {method}")

    return work, col_map
