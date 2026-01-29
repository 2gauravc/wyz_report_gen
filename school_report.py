#!/usr/bin/env python3
"""
Generate a School Summary HTML report from a class fitness CSV,
with dropdown filters (Grade, Class, Section, Gender) in the HTML.

Usage:
  python generate_school_summary.py --csv class_report.csv --out school_summary.html
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import jinja2


DEFAULT_TEMPLATE_NAME = "school_summary_template.html"


def _normalize_gender(x: Any) -> str:
    s = str(x or "").strip().lower()
    if s.startswith("b"):
        return "Boy"
    if s.startswith("g"):
        return "Girl"
    return "Unknown"


def _to_num(s: Any) -> float:
    try:
        if pd.isna(s):
            return float("nan")
        t = str(s).strip()
        if t == "" or t.lower() in {"na", "null", "none"}:
            return float("nan")
        return float(t)
    except Exception:
        return float("nan")


def _compute_bmi(height_cm: float, weight_kg: float) -> float:
    if not np.isfinite(height_cm) or not np.isfinite(weight_kg) or height_cm <= 0:
        return float("nan")
    m = height_cm / 100.0
    bmi = weight_kg / (m * m)
    return bmi if np.isfinite(bmi) else float("nan")


def _histogram(values: List[float], bins: int = 8) -> Dict[str, Any]:
    xs = np.array([v for v in values if np.isfinite(v)], dtype=float)
    if xs.size == 0:
        return {"labels": [], "counts": []}

    vmin = float(xs.min())
    vmax = float(xs.max())
    if vmin == vmax:
        return {"labels": [str(int(round(vmin)))], "counts": [int(xs.size)]}

    counts, edges = np.histogram(xs, bins=bins)
    labels = [f"{edges[i]:.0f}–{edges[i+1]:.0f}" for i in range(len(edges) - 1)]
    return {"labels": labels, "counts": counts.astype(int).tolist()}


def _pct(xs: List[float], p: float) -> float:
    arr = np.array([v for v in xs if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(np.percentile(arr, p))


def _mean(xs: List[float]) -> float:
    arr = np.array([v for v in xs if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan")
    return float(arr.mean())


def _median(xs: List[float]) -> float:
    return _pct(xs, 50)


def compute_group_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    heights = [_to_num(r.get("height")) for r in rows]
    weights = [_to_num(r.get("weight")) for r in rows]
    fitness = [_to_num(r.get("fitnessscore")) for r in rows]
    bmis = [_compute_bmi(_to_num(r.get("height")), _to_num(r.get("weight"))) for r in rows]

    # "Missing any test" based on *_best fields (same logic as before)
    test_cols = [
        "chairsquat_best",
        "ohmbp_best",
        "sbj_best",
        "smbt_best",
        "runspeed_best",
        "proagility_best",
        "slr_best",
        "shouldermob_best",
        "baleyesopen_best",
    ]

    def missing_any(r: Dict[str, Any]) -> bool:
        for c in test_cols:
            if c in r:
                if not np.isfinite(_to_num(r.get(c))):
                    return True
        return False

    miss_pct = (sum(1 for r in rows if missing_any(r)) / len(rows) * 100.0) if rows else float("nan")

    return {
        "n": len(rows),
        "avg_height": _mean(heights),
        "med_height": _median(heights),
        "p75_height": _pct(heights, 75),
        "avg_weight": _mean(weights),
        "med_weight": _median(weights),
        "p75_weight": _pct(weights, 75),
        "avg_bmi": _mean(bmis),
        "med_bmi": _median(bmis),
        "p75_bmi": _pct(bmis, 75),
        "avg_fitness": _mean(fitness),
        "med_fitness": _median(fitness),
        "p75_fitness": _pct(fitness, 75),
        "pct_missing_any": miss_pct,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Input CSV file path")
    ap.add_argument("--out", required=True, help="Output HTML file path")
    ap.add_argument("--template", default=DEFAULT_TEMPLATE_NAME, help="Template HTML path")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    template_path = Path(args.template)
    out_path = Path(args.out)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not template_path.exists():
        raise FileNotFoundError(f"Template not found: {template_path}")

    df = pd.read_csv(csv_path)

    # Convert to list of dict rows and add normalized gender
    raw_rows: List[Dict[str, Any]] = df.to_dict(orient="records")
    rows: List[Dict[str, Any]] = []
    for r in raw_rows:
        rr = {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in r.items()}
        rr["gender_norm"] = _normalize_gender(rr.get("gender"))
        rows.append(rr)

    # Meta sets for dropdown options
    grades = sorted({str(r.get("grade")).strip() for r in rows if r.get("grade") is not None and str(r.get("grade")).strip() != ""})
    classes = sorted({str(r.get("class")).strip() for r in rows if r.get("class") is not None and str(r.get("class")).strip() != ""})
    sections = sorted({str(r.get("section")).strip() for r in rows if r.get("section") is not None and str(r.get("section")).strip() != ""})

    # Precompute "All students" initial stats and initial charts (so the page loads with content)
    stats_all = compute_group_stats(rows)

    # initial gender split for default view
    boys = [r for r in rows if r.get("gender_norm") == "Boy"]
    girls = [r for r in rows if r.get("gender_norm") == "Girl"]
    stats_boys = compute_group_stats(boys)
    stats_girls = compute_group_stats(girls)

    context = {
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "grades": grades,
        "classes": classes,
        "sections": sections,
        # embed all rows for client-side filtering
        "rows_json": json.dumps(rows, ensure_ascii=False),
        # initial stats (all/boys/girls)
        "stats_json": json.dumps({"all": stats_all, "boys": stats_boys, "girls": stats_girls}, ensure_ascii=False),
    }

    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(str(template_path.parent)),
        autoescape=jinja2.select_autoescape(["html", "xml"]),
    )
    template = env.get_template(template_path.name)
    html = template.render(**context)

    out_path.write_text(html, encoding="utf-8")
    print(f"✅ Wrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
