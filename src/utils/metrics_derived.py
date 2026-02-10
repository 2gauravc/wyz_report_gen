# utils/metrics_derived.py

from __future__ import annotations
import pandas as pd


def compute_derived_metric(
    work: pd.DataFrame,
    metric_id: str,
    derived_spec: dict
) -> pd.Series:
    """
    Compute a derived level0 metric (e.g. min(left,right)).
    """
    method = (derived_spec.get("method") or "").lower()
    inputs = (
        derived_spec.get("inputs_from_dataset")
        or derived_spec.get("inputs")
        or []
    )

    if not inputs:
        raise ValueError(f"{metric_id}: derived metric has no inputs")

    missing = [c for c in inputs if c not in work.columns]
    if missing:
        raise ValueError(
            f"{metric_id}: missing derived inputs in work df: {missing}"
        )

    if method == "min":
        return work[inputs].min(axis=1, skipna=True)

    if method == "max":
        return work[inputs].max(axis=1, skipna=True)

    raise ValueError(
        f"{metric_id}: unsupported derived method '{method}'"
    )


def apply_level0_derived_metrics(
    work: pd.DataFrame,
    cfg: dict
) -> pd.DataFrame:
    """
    Applies all level0 derived metrics defined in YAML.
    """
    level0 = cfg.get("metrics", {}).get("level0", {})

    for metric_id, spec in level0.items():
        derived = spec.get("derived")
        if not isinstance(derived, dict):
            continue

        work[metric_id] = compute_derived_metric(
            work=work,
            metric_id=metric_id,
            derived_spec=derived
        )

    return work
