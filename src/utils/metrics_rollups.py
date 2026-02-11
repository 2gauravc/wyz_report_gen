from __future__ import annotations
from typing import List, Dict, Any
import pandas as pd
import numpy as np


def _round_clamp(series: pd.Series, ndigits: int = 1, min_v: float = 1, max_v: float = 5) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    s = s.round(ndigits)
    s = s.clip(lower=min_v, upper=max_v)
    return s


def _rollup_mean(work: pd.DataFrame, cols: List[str]) -> pd.Series:
    # row-wise mean ignoring NaN
    return work[cols].astype("float").mean(axis=1, skipna=True)


def apply_level1_rollups(work: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Builds Level1 domains from Level0 metric scores.

    For each domain in cfg["metrics"]["level1"]:
      - creates work[domain] = mean(child_metric_scores) rounded
      - creates work[f"{domain}_score"] using rollup.score.method (round_clamp)

    Assumptions:
      - Level0 scores exist: work[f"{metric_id}_score"]
    """
    level1 = (cfg.get("metrics", {}) or {}).get("level1", {}) or {}

    for domain_id, spec in level1.items():
        children = spec.get("child_metrics") or spec.get("children") or []
        if not children:
            continue

        rollup = spec.get("rollup", {}) or {}
        input_kind = (rollup.get("input") or "score").lower()  # "score" expected
        method = (rollup.get("method") or "mean").lower()
        ndigits = int(rollup.get("round", 1))

        # Build the list of child columns
        if input_kind == "score":
            child_cols = [f"{m}_score" for m in children]
        else:
            # If later you support input: "value"
            child_cols = list(children)

        missing = [c for c in child_cols if c not in work.columns]
        if missing:
            raise ValueError(f"Level1 '{domain_id}': missing child columns: {missing}")

        # Compute rollup value
        if method == "mean":
            val = _rollup_mean(work, child_cols)
        else:
            raise ValueError(f"Level1 '{domain_id}': unsupported rollup method '{method}'")

        # Store the "value" (rounded)
        work[domain_id] = pd.to_numeric(val, errors="coerce").round(ndigits)

        # Compute rollup score
        score_cfg = rollup.get("score", {}) or {}
        score_method = (score_cfg.get("method") or "round_clamp").lower()

        if score_method == "round_clamp":
            mn = float(score_cfg.get("min", 1))
            mx = float(score_cfg.get("max", 5))
            # for domain score, usually round to nearest int and clamp
            # but your YAML says round: 1 at rollup level; keep it consistent:
            domain_score = _round_clamp(work[domain_id], ndigits=ndigits, min_v=mn, max_v=mx)

            # Often you want integer domain score. If yes, uncomment next line:
            domain_score = domain_score.round(0).astype("Int64")

            work[f"{domain_id}_score"] = domain_score
        else:
            raise ValueError(f"Level1 '{domain_id}': unsupported score method '{score_method}'")

    return work
def apply_level2_rollups(work: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Builds Level2 constructs from Level1 domain scores.
    Creates:
      - work[foundation], work[foundation_score]
      - work[movement_skills], work[movement_skills_score]
    """
    level2 = (cfg.get("metrics", {}) or {}).get("level2", {}) or {}

    # only process actual metrics; skip the "views" key if present
    for metric_id, spec in level2.items():
        if metric_id == "views":
            continue

        children = spec.get("child_metrics") or spec.get("children") or []
        if not children:
            continue

        rollup = spec.get("rollup", {}) or {}
        input_kind = (rollup.get("input") or "score").lower()
        method = (rollup.get("method") or "mean").lower()
        ndigits = int(rollup.get("round", 1))

        # Level2 references Level1 ids like "strength"
        if input_kind == "score":
            child_cols = [f"{c}_score" for c in children]
        else:
            child_cols = list(children)

        missing = [c for c in child_cols if c not in work.columns]
        if missing:
            raise ValueError(f"Level2 '{metric_id}': missing child columns: {missing}")

        if method == "mean":
            val = _rollup_mean(work, child_cols)
        else:
            raise ValueError(f"Level2 '{metric_id}': unsupported rollup method '{method}'")

        work[metric_id] = pd.to_numeric(val, errors="coerce").round(ndigits)

        score_cfg = rollup.get("score", {}) or {}
        score_method = (score_cfg.get("method") or "round_clamp").lower()
        if score_method == "round_clamp":
            mn = float(score_cfg.get("min", 1))
            mx = float(score_cfg.get("max", 5))
            domain_score = _round_clamp(work[metric_id], ndigits=ndigits, min_v=mn, max_v=mx)
            domain_score = domain_score.round(0).astype("Int64")
            work[f"{metric_id}_score"] = domain_score
        else:
            raise ValueError(f"Level2 '{metric_id}': unsupported score method '{score_method}'")

    return work


def apply_level2_views(work: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Applies Level2 "views" such as grid_2x2 classification.
    Produces a categorical id column:
      - work["level2_grid_2x2"]
    """
    level2 = (cfg.get("metrics", {}) or {}).get("level2", {}) or {}
    views = level2.get("views", {}) or {}
    grid = views.get("grid_2x2", {}) or {}

    if not grid:
        return work

    axes = grid.get("axes", {}) or {}
    x_metric = ((axes.get("x") or {}).get("metric"))  # e.g. "foundation"
    y_metric = ((axes.get("y") or {}).get("metric"))  # e.g. "movement_skills"

    if not x_metric or not y_metric:
        raise ValueError("level2.views.grid_2x2.axes must define x.metric and y.metric")

    # We classify on SCORE (not raw)
    x_col = f"{x_metric}_score"
    y_col = f"{y_metric}_score"
    if x_col not in work.columns or y_col not in work.columns:
        raise ValueError(f"grid_2x2 requires {x_col} and {y_col} columns to exist")

    threshold = (grid.get("threshold", {}) or {})
    high_min = float(threshold.get("high_min", 4))

    x_high = pd.to_numeric(work[x_col], errors="coerce") >= high_min
    y_high = pd.to_numeric(work[y_col], errors="coerce") >= high_min

    # quadrant ids exactly as in YAML
    q_ll = "foundation_low_movement_low"
    q_hl = "foundation_high_movement_low"
    q_lh = "foundation_low_movement_high"
    q_hh = "foundation_high_movement_high"

    quad = pd.Series(pd.NA, index=work.index, dtype="string")

    quad.loc[(~x_high) & (~y_high)] = q_ll
    quad.loc[( x_high) & (~y_high)] = q_hl
    quad.loc[(~x_high) & ( y_high)] = q_lh
    quad.loc[( x_high) & ( y_high)] = q_hh

    work["level2_grid_2x2"] = quad
    return work


def apply_level3_rollups(work: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Builds Level3 overall score from Level1 domain scores.
    Creates:
      - work["overall"], work["overall_score"]
    """
    level3 = (cfg.get("metrics", {}) or {}).get("level3", {}) or {}
    overall_spec = level3.get("overall")
    if not isinstance(overall_spec, dict):
        return work

    children = overall_spec.get("child_metrics") or overall_spec.get("children") or []
    if not children:
        return work

    rollup = overall_spec.get("rollup", {}) or {}
    input_kind = (rollup.get("input") or "score").lower()
    method = (rollup.get("method") or "mean").lower()
    ndigits = int(rollup.get("round", 1))

    if input_kind == "score":
        child_cols = [f"{c}_score" for c in children]
    else:
        child_cols = list(children)

    missing = [c for c in child_cols if c not in work.columns]
    if missing:
        raise ValueError(f"Level3 'overall': missing child columns: {missing}")

    if method == "mean":
        val = _rollup_mean(work, child_cols)
    else:
        raise ValueError(f"Level3 'overall': unsupported rollup method '{method}'")

    work["overall"] = pd.to_numeric(val, errors="coerce").round(ndigits)

    score_cfg = rollup.get("score", {}) or {}
    score_method = (score_cfg.get("method") or "round_clamp").lower()
    if score_method == "round_clamp":
        mn = float(score_cfg.get("min", 1))
        mx = float(score_cfg.get("max", 5))
        overall_score = _round_clamp(work["overall"], ndigits=ndigits, min_v=mn, max_v=mx)
        overall_score = overall_score.round(0).astype("Int64")
        work["overall_score"] = overall_score
    else:
        raise ValueError(f"Level3 'overall': unsupported score method '{score_method}'")

    return work
