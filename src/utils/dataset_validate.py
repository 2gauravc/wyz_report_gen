from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any
import pandas as pd
import numpy as np

@dataclass
class ValidationReport:
    missing_required: List[str]
    missing_optional: List[str]
    high_nan_numeric: List[Tuple[str, int, int, float]]      # (field, nan_count, total, frac)
    out_of_range: List[Tuple[str, float | None, float | None, int]]  # (field, min, max, bad_count)

    def ok(self) -> bool:
        return len(self.missing_required) == 0

def validate_dataset_columns(cfg: dict, col_map: dict, work: pd.DataFrame) -> ValidationReport:
    dataset_cfg = cfg.get("dataset", {})
    sections = ("entity_columns", "measurement_columns")

    missing_required = []
    missing_optional = []

    for section in sections:
        section_cfg = dataset_cfg.get(section, {}) or {}
        for field, spec in section_cfg.items():
            required = bool(spec.get("required", False))
            src = col_map.get(field)

            is_derived = isinstance(spec.get("derive"), dict)
            in_work = field in work.columns and work[field].notna().any()

            # If it is derived, don't require a source column:
            if is_derived:
                # only flag if the derived result isn't present / all NA
                if required and not in_work:
                    missing_required.append(field)
                elif (not required) and not in_work:
                    # optional derived missing: you can choose to ignore or warn.
                    missing_optional.append(field)
                continue

            # Non-derived fields must have a source column
            if src is None:
                if required:
                    missing_required.append(field)
                else:
                    missing_optional.append(field)

    return ValidationReport(
        missing_required=missing_required,
        missing_optional=missing_optional,
        high_nan_numeric=[],
        out_of_range=[],
    )


def validate_values_basic(work: pd.DataFrame, cfg: dict, report: ValidationReport) -> ValidationReport:
    dataset_cfg = cfg.get("dataset", {})
    sections = ("entity_columns", "measurement_columns")

    for section in sections:
        section_cfg = dataset_cfg.get(section, {}) or {}
        for field, spec in section_cfg.items():
            if field not in work.columns:
                continue

            dtype = (spec.get("dtype") or "").lower()
            validate = spec.get("validate") or {}

            s = work[field]

            # numeric sanity: if too many NaN after coercion
            if dtype == "number":
                s_num = pd.to_numeric(s, errors="coerce")
                nan_count = int(s_num.isna().sum())
                total = int(len(s_num))
                frac = (nan_count / total) if total else 0.0

                # Flag only if it looks suspicious and the field is not optional-missing
                # (you can tune threshold)
                if total > 0 and frac > 0.30:
                    report.high_nan_numeric.append((field, nan_count, total, frac))

                # range checks
                if isinstance(validate, dict) and ("min" in validate or "max" in validate):
                    bad = pd.Series(False, index=s_num.index)
                    mn = validate.get("min", None)
                    mx = validate.get("max", None)

                    if mn is not None:
                        bad |= s_num < float(mn)
                    if mx is not None:
                        bad |= s_num > float(mx)

                    bad_count = int(bad.sum())
                    if bad_count > 0:
                        report.out_of_range.append((field, mn, mx, bad_count))

    return report

def print_validation_report(report: ValidationReport) -> None:
    if report.missing_required:
        print("❌ Missing REQUIRED columns:", ", ".join(report.missing_required))
    else:
        print("✅ No missing required columns")

    if report.missing_optional:
        print("⚠️ Missing optional columns:", ", ".join(report.missing_optional))

    if report.high_nan_numeric:
        print("⚠️ High NaN after numeric coercion (>30%):")
        for field, nan_count, total, frac in report.high_nan_numeric:
            pct = round(frac * 100, 1)
            print(f"   - {field}: {nan_count}/{total} NaN ({pct}%)")

    if report.out_of_range:
        print("⚠️ Out-of-range values:")
        for field, mn, mx, bad_count in report.out_of_range:
            print(f"   - {field}: {bad_count} rows outside [{mn}, {mx}]")
