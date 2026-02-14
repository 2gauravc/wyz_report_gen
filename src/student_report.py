import argparse
import re
import sys
from typing import Optional

import numpy as np
import pandas as pd

from utils.paths import resolve_repo
from utils.filenames import generate_report_filename
from utils.columns import apply_level0_value_columns, find_column
from utils.dataframe import safe_numeric_col
from utils.scoring import assign_scores
from render.html import render_html_for_first_n 
from render.pdf import convert_htmls_to_pdfs
from utils.yamlops import load_yaml_config
from utils.dataset import build_work_df_from_config
from utils.dataset_validate import (
    validate_dataset_columns,
    validate_values_basic,
    print_validation_report,
)
from utils.columns import apply_level0_value_columns
from utils.metrics_derived import apply_level0_derived_metrics

from utils.aggregates import apply_level0_aggregates
from utils.metrics_scoring import apply_level0_scores
from utils.metrics_rollups import (
    apply_level1_rollups,
    apply_level2_rollups,
    apply_level2_views,
    apply_level3_rollups,
)
from utils.aggregates import apply_all_aggregates


def add_group_stats(
    df,
    value_col,
    group_cols,
    score_fn=None,
    better: str = "higher",  # "higher" or "lower"
    percentile: int = 80
):
    grp = df.groupby(group_cols)[value_col]

    # average
    df[f"{value_col}_avg"] = grp.transform("mean").round(1)

    # percentile logic
    p = percentile if better == "higher" else 100 - percentile

    df[f"{value_col}_p"] = grp.transform(
        lambda x: (
            np.nanpercentile(x.dropna(), p)
            if x.dropna().size > 0
            else np.nan
        )
    ).round()
    #print ("added percentile") #add value also 
    # best
    best_fn = "max" if better == "higher" else "min"
    df[f"{value_col}_best"] = grp.transform(best_fn)

    # score
    if score_fn is not None:
        scores = grp.transform(score_fn).round(0) 
        if better == "lower":
            max_score = scores.max()
            scores = max_score - scores + 1

        df[f"{value_col}_score"] = scores


def generate_csv(input_csv: str, output_csv: str, config_path:str) -> pd.DataFrame:
    cfg = load_yaml_config(config_path)
    df = pd.read_csv(input_csv)

    # Step: build dataset (entity + measurements + derived_attributes)
    work, col_map = build_work_df_from_config(df, cfg)

    report = validate_dataset_columns(cfg, col_map, work)
    report = validate_values_basic(work, cfg, report)
    print_validation_report(report)

    if not report.ok():
        raise ValueError("Required columns missing; cannot continue.")
    
    # NEW: create canonical level0 value columns from dataset
    
    work = apply_level0_value_columns(work, cfg)

    # existing: derived metrics (straight_leg_raise, etc.)
    work = apply_level0_derived_metrics(work, cfg)

    # Scoring
    work = apply_level0_scores(work, cfg)

    # Rollups
    work = apply_level1_rollups(work, cfg)
    work = apply_level2_rollups(work, cfg)
    work = apply_level2_views(work, cfg)
    work = apply_level3_rollups(work, cfg)

    # Total fitness score
    level0_ids = list(cfg["metrics"]["level0"].keys())
    score_cols = [f"{m}_score" for m in level0_ids if f"{m}_score" in work.columns]
    work["fitnessscore"] = work[score_cols].sum(axis=1, min_count=1).astype("Int64")
    work["maxfitnessscore"] = len(score_cols) * 5
    
    # ✅ Aggregates (Level0 + Level1 + Level2 + Level3, depending on YAML)
    work = apply_all_aggregates(work, cfg)
    
    print("Number of Columns in CSV generated:", len(work.columns))
    # For now, just write raw + derived measurements
    work.to_csv(output_csv, index=False)

    
    return work

def main(argv: Optional[list] = None):
    parser = argparse.ArgumentParser(
    description='Generate WYRDZ Fitness student reports.')
    parser.add_argument(
        '--run-config',
        default='config/report_run_config.yaml',
        help='Run configuration file')
    parser.add_argument(
    '--app-config',
    default='config/app_config.yaml',
    help='Application configuration (PDF engine, system settings)')
    
    args = parser.parse_args(argv)
    run_cfg = load_yaml_config(args.run_config)
    input_csv_file = resolve_repo(run_cfg["input_data"]["file"])
    metrics_yaml_file = resolve_repo(run_cfg["metrics"]["metrics_config_yaml"])
    output_csv_path = resolve_repo(run_cfg["outputs"]["csv_outdir"])
    output_csv_file = generate_report_filename(output_csv_path, "report_output", ".csv")
    outputs = run_cfg.get("outputs", {})
    is_render_html = outputs.get("render_html", False)
    is_to_pdf = outputs.get("to_pdf", False)

    html_outdir = resolve_repo(run_cfg["outputs"]["html_outdir"])
    pdf_outdir = resolve_repo(run_cfg["outputs"]["pdf_outdir"])
    html_template = run_cfg["templates"]["student"]
    limit = run_cfg["outputs"]["limit"]
    
    app_cfg = load_yaml_config(args.app_config)

    try:
        df_out = generate_csv(input_csv_file, output_csv_file, metrics_yaml_file)
        print(f"Wrote report to {output_csv_file}")
        if is_render_html:
            render_html_for_first_n(df_out, html_template, html_outdir, metrics_yaml_file, limit)
            print(f"Rendered HTML reports to {html_outdir}")
        if is_to_pdf:
            # convert HTML files in outdir to PDF using wkhtmltopdf
            convert_htmls_to_pdfs(html_outdir, pdf_outdir)
        
    except FileNotFoundError:
        print("error in try")
        sys.exit(2)


if __name__ == '__main__':
    main()