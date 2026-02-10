from __future__ import annotations
import pandas as pd

def apply_level0_value_columns(work: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Creates work[<metric_id>] for metrics that have input.from_dataset,
    by copying the dataset measurement column into the metric_id column.
    Derived metrics are handled elsewhere (keep as-is).
    """
    level0 = (cfg.get("metrics", {}) or {}).get("level0", {}) or {}

    for metric_id, spec in level0.items():
        # Skip if this metric is derived (computed later/elsewhere)
        if isinstance(spec.get("derived"), dict):
            continue

        inp = spec.get("input") or {}
        src = inp.get("from_dataset")
        if not src:
            continue

        if src not in work.columns:
            raise ValueError(f"{metric_id}: input.from_dataset '{src}' not found in work dataframe")

        # Create a canonical metric value column
        work[metric_id] = work[src]

    return work
