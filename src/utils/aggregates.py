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
