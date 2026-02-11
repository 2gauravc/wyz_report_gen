import argparse
import re
import sys
from typing import Optional

import numpy as np
import pandas as pd
import os
import jinja2
import subprocess
import shutil
import glob

from pathlib import Path
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
from utils.aggregates import apply_level0_aggregates
from utils.metrics_scoring import apply_level0_scores
from utils.metrics_rollups import apply_level1_rollups
from utils.metrics_rollups import (
    apply_level1_rollups,
    apply_level2_rollups,
    apply_level2_views,
    apply_level3_rollups,
)

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
    from utils.columns import apply_level0_value_columns
    from utils.metrics_derived import apply_level0_derived_metrics
    work = apply_level0_value_columns(work, cfg)

    # existing: derived metrics (straight_leg_raise, etc.)
    work = apply_level0_derived_metrics(work, cfg)

    # NOW aggregates will find the metric_id columns
    work = apply_level0_aggregates(work, cfg)

    # Scoring 
    work = apply_level0_scores(work, cfg)

    # then:
    work = apply_level1_rollups(work, cfg)
    
    # level2 + views
    work = apply_level2_rollups(work, cfg)
    work = apply_level2_views(work, cfg)

    # level3
    work = apply_level3_rollups(work, cfg)
    
    # For now, just write raw + derived measurements
    work.to_csv(output_csv, index=False)
    
    return work

def main(argv: Optional[list] = None):
    parser = argparse.ArgumentParser(description='Generate ChairSquats1min report per student.')
    parser.add_argument('input', nargs='?', default='input_data/master_6th_class.csv', help='Input CSV file')
    parser.add_argument('-o', '--output', default='out_csv/report_output.csv', help='Output CSV file')
    parser.add_argument('--render-html', action='store_true', help='Render HTML reports for first N students using template')
    parser.add_argument('--template', default='report_template.html', help='HTML template path')
    parser.add_argument('--html-outdir', default='out_html', help='Output directory for generated HTML files')
    parser.add_argument('--pdf-outdir', default='out_pdf', help='Output directory for generated PDF files')
    parser.add_argument('--to-pdf', action='store_true', help='Convert generated HTML reports to PDF using wkhtmltopdf')
    parser.add_argument('--limit', type=int, default=10, help='Number of students to render')
    parser.add_argument('--config', default='config/metrics/fitness_metrics.yaml', help='YAML config file')


    args = parser.parse_args(argv)

    try:
        df_out = generate_csv(args.input, args.output, args.config)
        print(f"Wrote report to {args.output}")
        #if args.render_html:
        #    render_html_for_first_n(df_out, args.template, args.html_outdir, args.limit)
        #    print(f"Rendered HTML reports to {args.html_outdir}")
        #if args.to_pdf:
        #    # convert HTML files in outdir to PDF using wkhtmltopdf
        #    try:
        #        convert_htmls_to_pdfs(args.html_outdir, args.pdf_outdir)
        #    except FileNotFoundError as e:
        #        print(str(e), file=sys.stderr)
        #        sys.exit(3)
    except FileNotFoundError:
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()