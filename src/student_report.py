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
from utils.columns import find_column
from utils.dataframe import safe_numeric_col
from utils.scoring import assign_scores
from render.html import render_html_for_first_n 
from render.pdf import convert_htmls_to_pdfs

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


def process(input_csv: str, output_csv: str) -> pd.DataFrame:
    df = pd.read_csv(input_csv)
    
    # Build working DataFrame with standard column names
    work = pd.DataFrame()

    # Identify columns for report - with some flexibility in col names and build 'work' dataframe
    ## Student metadata columns
    sr_col = find_column(df.columns, ["sr"]) 
    name_col = find_column(df.columns, ["name"]) 
    class_col = find_column(df.columns, ["class"])
    
    work["srno"] = df[sr_col] if sr_col in df.columns else pd.RangeIndex(start=1, stop=len(df) + 1)
    work["name"] = df[name_col] if name_col in df.columns else np.nan
    work["class"] = df[class_col] if class_col in df.columns else np.nan
    ### split class into grade and section. Grade is the number and section is the character after the number
    work["grade"] = work["class"].str.extract(r'(\d+)')
    work["section"] = work["class"].str.extract(r'([A-Za-z])$')
   
    ## Student basic data columns
    gender_col = find_column(df.columns, ["bg", "gender"])
    height_col = find_column(df.columns, ["height"]) # in cm
    weight_col = find_column(df.columns, ["weight"]) # in kg

    work["gender"] = df[gender_col] if gender_col in df.columns else np.nan
    work['height'] = safe_numeric_col(df, height_col)
    work['weight'] = safe_numeric_col(df, weight_col)

    ## Strength Metrics columns
    chairsquat_col = find_column(df.columns, ["chairsquat", "chairsquats1min"])
    ohmbp_col = find_column(df.columns, ["ohmbp", "ohmbseatedpress1min"])

    work["chairsquat"] = safe_numeric_col(df, chairsquat_col)
    work["ohmbp"] = safe_numeric_col(df, ohmbp_col)

    ## Power Metrics columns
    sbj_col = find_column(df.columns, ["sbj", "standingbroadjump"])
    smbt_col = find_column(df.columns, ["smbt","sittingmedicineballthrow"])

    work["sbj"] = safe_numeric_col(df, sbj_col)
    work["smbt"] = safe_numeric_col(df, smbt_col)

    ## Speed & Agility Metrics columns
    runspeed_col = find_column(df.columns, ["runspeed", "speed20mbest"])
    proagility_col = find_column(df.columns, ["proagility"])

    work["runspeed"] = safe_numeric_col(df, runspeed_col)
    work["proagility"] = safe_numeric_col(df, proagility_col)

    ## Flexibility Metrics columns
    slr_left_col = find_column(df.columns, ["slrleft", "straightlegraiseleft"]) 
    slr_right_col = find_column(df.columns, ["slright", "straightlegraiseright"])

    work["slr_left"] = safe_numeric_col(df, slr_left_col)
    work["slr_right"] = safe_numeric_col(df, slr_right_col)
    
    shouldermob_left_col = find_column(df.columns, ["shouldermobleft", "shouldermobilityleft"])
    shouldermob_right_col = find_column(df.columns, ["shouldermobright", "shouldermobilityright"])

    work["shouldermob_left"] = safe_numeric_col(df, shouldermob_left_col)
    work["shouldermob_right"] = safe_numeric_col(df, shouldermob_right_col)
        
    ## Balance metrics columns
    baleyesopen_left_col = find_column(df.columns, ["baleyesopenleft", "slbalanceeoleft"])
    baleyesopen_right_col = find_column(df.columns, ["baleyesopenright", "slbalanceeoright"])  
    
    work["baleyesopen_left"] = safe_numeric_col(df, baleyesopen_left_col)
    work["baleyesopen_right"] = safe_numeric_col(df, baleyesopen_right_col)
    
    
    ## Comments column
    comments_col = find_column(df.columns, ["comments", "notes", "remark", "remarks"]) or None
    work["comments"] = (
        df[comments_col].astype("string")
        if comments_col is not None and comments_col in df.columns
        else pd.Series(pd.NA, index=df.index, dtype="string")
    )



    ## Find how many columns were not found (None) 
    expected_cols = {
    "sr": sr_col,
    "name": name_col,
    "class": class_col,
    "gender": gender_col,
    "height": height_col,
    "weight": weight_col,
    "chairsquat": chairsquat_col,
    "ohmbp": ohmbp_col,
    "sbj": sbj_col,
    "smbt": smbt_col,
    "runspeed": runspeed_col,
    "proagility": proagility_col,
    "slr_left": slr_left_col,
    "slr_right": slr_right_col,
    "shouldermob_left": shouldermob_left_col,
    "shouldermob_right": shouldermob_right_col,
    "baleyesopen_left": baleyesopen_left_col,
    "baleyesopen_right": baleyesopen_right_col,
    "comments": comments_col,
    }


    missing_cols = [name for name, col in expected_cols.items() if col is None]
    missing_count = len(missing_cols)
    if missing_cols:
        print(
            f"Missing columns ({missing_count}): "
            + ", ".join(missing_cols)
            )
    else:
        print("No missing columns 🎉")

    
    # Compute derived statistics 
    
    ## BMI 
    work["bmi"] = (work["weight"] / ((work["height"] / 100) ** 2)).round(1)

    ## SLR 
    work["slr"] = work[["slr_left", "slr_right"]].min(axis=1)

    ## Shoulder Mobility 
    work["shouldermob"] = work[["shouldermob_left", "shouldermob_right"]].min(axis=1)

    ## Balance - Eyes Open
    work["baleyesopen"] = work[["baleyesopen_left", "baleyesopen_right"]].min(axis=1)
    
    # Compute Aggregate stats 

    ## ChairSquats
    add_group_stats(
    work,
    value_col="chairsquat",
    group_cols=["grade", "gender"],
    score_fn=lambda x: assign_scores(x, bins_count=5),
    better="higher",
    percentile=80)

    ## Overhead Medicine Ball Press (OHMBP) 
    add_group_stats(
    work,
    value_col="ohmbp",
    group_cols=["grade", "gender"],
    score_fn=lambda x: assign_scores(x, bins_count=5),
    better="higher",
    percentile=80)
    
    ## Standing Broad Jump (SBJ)
    add_group_stats(
    work,
    value_col="sbj",
    group_cols=["grade", "gender"],
    score_fn=lambda x: assign_scores(x, bins_count=5),
    better="higher",
    percentile=80)

    ## Seated Medicine Ball Throw (SMBT)
    add_group_stats(
    work,
    value_col="smbt",
    group_cols=["grade", "gender"],
    score_fn=lambda x: assign_scores(x, bins_count=5),
    better="higher",
    percentile=80)

    ## Runspeed. Lower the better. Hence 20th is 80th percentile student 
    add_group_stats(
    work,
    value_col="runspeed",
    group_cols=["grade", "gender"],
    score_fn=lambda x: assign_scores(x, bins_count=5),
    better="lower",
    percentile=80)

    ## Proagility. Lower the better 
    add_group_stats(
    work,
    value_col="proagility",
    group_cols=["grade", "gender"],
    score_fn=lambda x: assign_scores(x, bins_count=5),
    better="lower",
    percentile=80)

    ## Straight Leg Raise (SLR). Ordinal scores (1,2,3)
    grp = work.groupby(["grade", "gender"])["slr"]
    work["slr_avg"] = grp.transform("mean").round(1)
    work["slr_pct_3"] = grp.transform(
    lambda x: (x == 3).mean() * 100).round(0)
    work["slr_best"] = grp.transform("max")
    score_map = {1: 1, 2: 3, 3: 5}
    work["slr_score"] = work["slr"].map(score_map)

    ## Shoulder Mobility. Ordinal scores (1,2,3)
    grp = work.groupby(["grade", "gender"])["shouldermob"]
    work["shouldermob_avg"] = grp.transform("mean").round(1)
    work["shouldermob_pct_3"] = grp.transform(
    lambda x: (x == 3).mean() * 100).round(0)
    score_map = {1: 1, 2: 3, 3: 5}
    
    work["shouldermob_best"] = grp.transform("max")
    work["shouldermob_score"] = work["shouldermob"].map(score_map)

    # Balance - Eyes Open
    add_group_stats(
    work,
    value_col="baleyesopen",
    group_cols=["grade", "gender"],
    score_fn=lambda x: pd.cut(
        x,
        bins=[0, 6, 12, 18, 24, 30],
        labels=[1, 2, 3, 4, 5],
        right=True
        ).astype("float"),
    better="higher",
    percentile=80)
        
    score_cols = ['chairsquat_score', 'ohmbp_score', 'sbj_score', 'smbt_score', 'runspeed_score', 'proagility_score', 'slr_score', 'shouldermob_score', 'baleyesopen_score']
    def sum_scores(row):
        s = 0
        count = 0
        for c in score_cols:
            if c in row and pd.notna(row[c]):
                s += float(row[c])
                count += 1
        return int(s) if count > 0 else np.nan

    work['fitnessscore'] = work.apply(sum_scores, axis=1).round(0).astype('Int64')    
    work['maxfitnessscore'] = sum([5 for c in score_cols if c in work.columns])

    # Reorder columns as requested (include many of the newly computed fields)
    out_cols = [
    "srno", "name", "class", "grade", "section", "gender", "height", "weight", 
    "bmi", "bmi_avg",
    "chairsquat", "chairsquat_avg", "chairsquat_p", "chairsquat_best", "chairsquat_score",  
    "ohmbp","ohmbp_avg", "ohmbp_p", "ohmbp_best", "ohmbp_score", 
    "sbj", "sbj_avg", "sbj_p", "sbj_best", "sbj_score", 
    "smbt","smbt_avg", "smbt_p", "smbt_best", "smbt_score",
    "runspeed", "runspeed_avg", "runspeed_p", "runspeed_best", "runspeed_score", 
    "proagility", "proagility_avg", "proagility_p", "proagility_best", "proagility_score",
    "slr", "slr_avg", "slr_pct_3", "slr_best", "slr_score", 
    "shouldermob", "shouldermob_avg", "shouldermob_pct_3", "shouldermob_best", "shouldermob_score", 
    "baleyesopen", "baleyesopen_avg", "baleyesopen_p", "baleyesopen_best", "baleyesopen_score",
    "fitnessscore", "maxfitnessscore",
    "comments"]
    
    # Some columns may not exist; filter to existing ones
    out = work[[c for c in out_cols if c in work.columns]]

    out.to_csv(output_csv, index=False)
    return out


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

    args = parser.parse_args(argv)

    try:
        df_out = process(args.input, args.output)
        print(f"Wrote report to {args.output}")
        if args.render_html:
            render_html_for_first_n(df_out, args.template, args.html_outdir, args.limit)
            print(f"Rendered HTML reports to {args.html_outdir}")
        if args.to_pdf:
            # convert HTML files in outdir to PDF using wkhtmltopdf
            try:
                convert_htmls_to_pdfs(args.html_outdir, args.pdf_outdir)
            except FileNotFoundError as e:
                print(str(e), file=sys.stderr)
                sys.exit(3)
    except FileNotFoundError:
        print(f"Input file not found: {args.input}", file=sys.stderr)
        sys.exit(2)


if __name__ == '__main__':
    main()