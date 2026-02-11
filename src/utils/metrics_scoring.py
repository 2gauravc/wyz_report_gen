from __future__ import annotations
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd


def _score_quantile_bins_group(s: pd.Series, bins: int) -> pd.Series:
    """
    Score within one group using quantile bins -> 1..bins.
    Robust fallback when group is too small or has too few unique values.
    """
    s_num = pd.to_numeric(s, errors="coerce")

    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    valid = s_num.dropna()
    if valid.empty:
        return out

    # If too few unique values, qcut can fail; fallback to rank-based bins.
    uniq = valid.nunique(dropna=True)
    if uniq < 2:
        # everyone same value -> middle score
        out.loc[valid.index] = int(np.ceil(bins / 2))
        return out

    try:
        # duplicates="drop" avoids failure when repeated quantiles collapse
        cats = pd.qcut(valid, q=bins, labels=False, duplicates="drop")
        # cats is 0..k-1; map to 1..k then stretch to 1..bins if k<bins
        k = int(cats.max() + 1) if len(cats) else 0
        if k <= 0:
            out.loc[valid.index] = int(np.ceil(bins / 2))
            return out
        scores = cats.astype("Int64") + 1  # 1..k
        if k != bins:
            # stretch 1..k to 1..bins
            scores = ((scores - 1) * (bins - 1) / (k - 1) + 1).round().astype("Int64")
        out.loc[valid.index] = scores
        return out
    except Exception:
        # fallback: rank then cut into bins
        r = valid.rank(method="average")  # 1..n
        n = len(r)
        if n == 1:
            out.loc[valid.index] = int(np.ceil(bins / 2))
            return out
        # convert rank to 1..bins
        scores = ((r - 1) * (bins - 1) / (n - 1) + 1).round().astype("Int64")
        out.loc[valid.index] = scores
        return out


def _score_fixed_bins(s: pd.Series, bins_spec: List[Dict[str, Any]]) -> pd.Series:
    """
    bins_spec: list of {min, max, score}. inclusive on both ends by default.
    """
    s_num = pd.to_numeric(s, errors="coerce")
    out = pd.Series(pd.NA, index=s.index, dtype="Int64")
    for b in bins_spec:
        mn = b.get("min", None)
        mx = b.get("max", None)
        sc = b.get("score", None)
        if sc is None:
            continue

        mask = pd.Series(True, index=s.index)
        if mn is not None:
            mask &= s_num >= float(mn)
        if mx is not None:
            mask &= s_num <= float(mx)

        out.loc[mask] = int(sc)
    return out


def _score_fixed_map(s: pd.Series, map_spec: Dict[Any, Any]) -> pd.Series:
    """
    map_spec keys may come in as ints or strings from YAML.
    We'll try both.
    """
    # normalize map keys to both str and int forms
    m: Dict[Any, Any] = {}
    for k, v in map_spec.items():
        m[k] = v
        try:
            m[int(k)] = v
        except Exception:
            pass
        m[str(k)] = v

    def _map_one(x):
        if pd.isna(x):
            return pd.NA
        if x in m:
            return m[x]
        # try numeric conversion
        try:
            xi = int(float(x))
            if xi in m:
                return m[xi]
        except Exception:
            pass
        # try string
        xs = str(x).strip()
        return m.get(xs, pd.NA)

    out = s.map(_map_one)
    return pd.Series(out, index=s.index, dtype="Int64")


def apply_level0_scores(work: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    For each metrics.level0.<metric_id>, compute:
      work[f"{metric_id}_score"]

    Uses spec.process.score:
      - quantile_bins (optionally group_by + bins)
      - fixed_bins
      - fixed_map
    Respects invert:true (flip score within 1..bins).
    """
    level0 = (cfg.get("metrics", {}) or {}).get("level0", {}) or {}

    for metric_id, spec in level0.items():
        if metric_id not in work.columns:
            # skip if not materialized
            continue

        process = spec.get("process", {}) or {}
        score_cfg = process.get("score", {}) or {}
        method = (score_cfg.get("method") or "").lower()
        invert = bool(score_cfg.get("invert", False))

        group_cols = process.get("group_by", []) or []
        s = work[metric_id]

        out_col = f"{metric_id}_score"

        if method == "quantile_bins":
            bins = int(score_cfg.get("bins", 5))

            if group_cols:
                scored = (
                    work.groupby(group_cols, dropna=False)[metric_id]
                    .transform(lambda g: _score_quantile_bins_group(g, bins=bins))
                )
            else:
                scored = _score_quantile_bins_group(s, bins=bins)

            scored = pd.Series(scored, index=work.index, dtype="Int64")

            if invert:
                # invert within 1..bins: new = (bins + 1 - old)
                scored = scored.apply(lambda x: (bins + 1 - x) if pd.notna(x) else pd.NA).astype("Int64")

            work[out_col] = scored

        elif method == "fixed_bins":
            bins_spec = score_cfg.get("bins", []) or []
            scored = _score_fixed_bins(s, bins_spec=bins_spec)
            work[out_col] = scored

        elif method == "fixed_map":
            map_spec = score_cfg.get("map", {}) or {}
            scored = _score_fixed_map(s, map_spec=map_spec)
            work[out_col] = scored

        elif method == "" or method is None:
            # no scoring configured
            continue

        else:
            raise ValueError(f"{metric_id}: unsupported score method '{method}'")

    return work
