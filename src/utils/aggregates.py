import numpy as np
import pandas as pd

def compute_aggregates(
    work: pd.DataFrame,
    metric_id: str,
    series: pd.Series,
    group_cols: list[str],
    aggregates: list[dict],
    direction: str,
) -> pd.DataFrame:
    grp = work.groupby(group_cols)[series.name]

    for agg in aggregates:
        name = agg["name"]

        if name == "avg":
            work[f"{metric_id}_avg"] = grp.transform("mean").round(1)

        elif name == "p80":
            q = agg.get("q", 80)
            p = q if direction == "higher_is_better" else 100 - q
            work[f"{metric_id}_p80"] = grp.transform(
                lambda x: np.nanpercentile(x.dropna(), p) if len(x.dropna()) else np.nan
            ).round(1)

        elif name == "best":
            if direction == "higher_is_better":
                work[f"{metric_id}_best"] = grp.transform("max")
            else:
                work[f"{metric_id}_best"] = grp.transform("min")

        else:
            raise ValueError(f"Unsupported aggregate: {name}")

    return work

def _resolve_aggregate_series(work: pd.DataFrame, metric_id: str, spec: dict) -> pd.Series | None:
    """
    Decide which column to aggregate for this metric:
      - metric_id (value)
      - metric_id_score (score)

    Priority:
      1) spec.process.input: "score" | "value"
      2) if spec has "rollup": default to "score"
      3) else default to "value"
    """
    process = spec.get("process", {}) or {}
    input_mode = (process.get("input") or "").strip().lower()

    if input_mode == "score":
        col = f"{metric_id}_score"
    elif input_mode == "value":
        col = metric_id
    else:
        # defaulting rule
        col = f"{metric_id}_score" if "rollup" in spec else metric_id

    if col not in work.columns:
        return None
    return work[col]


def apply_aggregates_for_level(work: pd.DataFrame, level_spec: dict) -> pd.DataFrame:
    """
    Apply aggregates for any level spec dict (level0/level1/level2/level3).
    Uses existing compute_aggregates().
    """
    for metric_id, spec in (level_spec or {}).items():
        process = spec.get("process", {}) or {}
        aggregates = process.get("aggregates", []) or []
        if not aggregates:
            continue

        group_cols = process.get("group_by", []) or []
        direction = (process.get("direction") or "higher_is_better").strip().lower()

        series = _resolve_aggregate_series(work, metric_id, spec)
        if series is None:
            # if you prefer strict behavior, raise here
            continue

        work = compute_aggregates(
            work=work,
            metric_id=metric_id,
            series=series,
            group_cols=group_cols,
            aggregates=aggregates,
            direction=direction,
        )

    return work


def apply_all_aggregates(work: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Apply aggregates for level0/1/2/3 if present in YAML.
    """
    metrics = (cfg.get("metrics", {}) or {})
    for level_name in ("level0", "level1", "level2", "level3"):
        level_spec = metrics.get(level_name, {}) or {}
        # level2 might include "views" which usually won't have process.aggregates; harmless
        work = apply_aggregates_for_level(work, level_spec)
    return work

# Unused Function 
def apply_level0_aggregates(work: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    level0 = cfg["metrics"]["level0"]

    for metric_id, spec in level0.items():
        process = spec.get("process", {})
        aggregates = process.get("aggregates", [])
        if not aggregates:
            continue

        group_cols = process.get("group_by", [])
        direction = process.get("direction", "higher_is_better")

        # metric values already exist in work
        if metric_id not in work.columns:
            continue

        work = compute_aggregates(
            work=work,
            metric_id=metric_id,
            series=work[metric_id],
            group_cols=group_cols,
            aggregates=aggregates,
            direction=direction,
        )

    return work
