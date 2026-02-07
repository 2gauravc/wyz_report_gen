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

def display_value(v, na="NA"):
    if v is None:
        return na
    if isinstance(v, float) and np.isnan(v):
        return na
    if pd.isna(v):
        return na
    return v

def safe_numeric_col(
    source_df: pd.DataFrame,
    col_name: str,
    default=np.nan
) -> pd.Series:
    """
    Safely extract a column and convert it to numeric.
    Returns a Series aligned to source_df.index.
    """
    if col_name in source_df.columns:
        return pd.to_numeric(source_df[col_name], errors="coerce")
    else:
        return pd.Series(default, index=source_df.index)

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

def _clean_col(col: str) -> str:
    return re.sub(r"[^a-z0-9]", "", col.lower())


def find_column(columns, patterns):
    cleaned = {col: _clean_col(col) for col in columns}
    for col, c in cleaned.items():
        for pat in patterns:
            if pat in c:
                return col
    return None


def assign_scores(s: pd.Series, bins_count: int = 5) -> pd.Series:
    # Assign scores 1..bins_count based on equal-width bins within group
    vals = s.copy()
    numeric = pd.to_numeric(vals, errors="coerce")
    if numeric.dropna().empty:
        return pd.Series([np.nan] * len(s), index=s.index)
    mn = numeric.min()
    mx = numeric.max()
    if mn == mx:
        return pd.Series([bins_count if not np.isnan(v) else np.nan for v in numeric], index=s.index)
    bins = np.linspace(mn, mx, bins_count + 1)
    labels = list(range(1, bins_count + 1))
    scored = pd.cut(numeric, bins=bins, labels=labels, include_lowest=True)
    return scored.astype(float)


def render_html_for_first_n(df_out: pd.DataFrame, template_path: str, out_dir: str, n: int = 10) -> None:
    """Render HTML reports for the first n students from the processed DataFrame.

    df_out: DataFrame returned by process() with columns used in template.
    template_path: path to `report_template.html`.
    out_dir: directory to write HTML files.
    n: number of students to render (first n rows).
    """
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")
    os.makedirs(out_dir, exist_ok=True)

    with open(template_path, 'r', encoding='utf-8') as f:
        tmpl_src = f.read()

    template = jinja2.Template(tmpl_src)

    def sanitize_fname(s: str) -> str:
        return re.sub(r"[^0-9A-Za-z._-]", "_", s)[:80]

    for i, row in df_out.head(n).iterrows():
        # map many possible column names to the template variables
        ctx = {
            'logo_url': "file:///workspaces/codespaces-blank/wdyrz/assets/images/wyrz_logo.svg",
            'school_name': row.get('schoolname', 'St Dominic Savio, Kanpur'),
            'name': row.get('name', ''),
            'height': row.get('height', ''),
            'fitness_score': row.get('fitnessscore', ''),
            'maxfitness_score': row.get('maxfitnessscore', ''),
            'class': row.get('class', ''),
            'gender': row.get('gender', ''),
            'weight': row.get('weight', ''),
            'date': (
                    row.get('date')
                    if pd.notna(row.get('date'))
                    else '01.01.2026'),

            # STRENGTH
            ## Chair squat
            'chairsquat_measure': row.get('chairsquat', ''),
            'chairsquat_avg': row.get('chairsquat_avg', ''),
            'chairsquat_p': row.get('chairsquat_p', ''),
            'chairsquat_best': row.get('chairsquat_best', ''),
            'chairsquat_score':row.get('chairsquat_score', ''),

            ## Overhead Medicine Ball Press 
            'ohmbp_measure': row.get('ohmbp', ''),
            'ohmbp_avg': row.get('ohmbp_avg', ''),
            'ohmbp_p': row.get('ohmbp_p', ''),
            'ohmbp_best': row.get('ohmbp_best', ''),
            'ohmbp_score':row.get('ohmbp_score', ''),

            # POWER 
            ## Standing Broad Jump 
            'sbj_measure': row.get('sbj', ''),
            'sbj_avg': row.get('sbj_avg', ''),
            'sbj_p': row.get('sbj_p', ''),
            'sbj_best': row.get('sbj_best', ''),
            'sbj_score': row.get('sbj_score', ''),

            ## Seated Medicine Ball Throw (SMBT)
            'smbt_measure': row.get('smbt', ''),
            'smbt_avg': row.get('smbt_avg', ''),
            'smbt_p': row.get('smbt_p', ''),
            'smbt_best': row.get('smbt_best', ''),
            'smbt_score': row.get('smbt_score', ''),

            # SPEED & AGILITY
            ## Run Speed
            'runspeed_measure': row.get('runspeed', ''),
            'runspeed_avg': row.get('runspeed_avg', ''),
            'runspeed_p':row.get('runspeed_p', ''),
            'runspeed_best': row.get('runspeed_best'),
            'runspeed_score': row.get('runspeed_score'),

            ## Pro Agility
            'proagility_measure': row.get('proagility', ''),
            'proagility_avg': row.get('proagility_avg', ''),
            'proagility_p': row.get('proagility_p',''),
            'proagility_best': row.get('proagility_best', ''),
            'proagility_score': row.get('proagility_score', ''),

            # FLEXIBILITY

            ## Straight Leg Raise (SLR)
            'slr_measure': row.get('slr', ''),
            'slr_avg': row.get('slr_avg', ''),
            'slr_pct_3': row.get('slr_pct_3' ,''),
            'slr_best': row.get('slr_best', ''),
            'slr_score': row.get('slr_score', ''),

            ## Shoulder Mobility
            'shouldermob_measure': row.get('shouldermob', ''),
            'shouldermob_avg': row.get('shouldermob_avg', ''),
            'shouldermob_pct_3': row.get('shouldermob_pct_3' ,''),
            'shouldermob_best': row.get('shouldermob_best', ''),
            'shouldermob_score': row.get('shouldermob_score', ''),

            # BALANCE 
            'baleyesopen_measure': row.get('baleyesopen', ''),
            'baleyesopen_avg': row.get('baleyesopen_avg', ''),
            'baleyesopen_p': row.get('baleyesopen_p', ''),
            'baleyesopen_best': row.get('baleyesopen_best', ''),
            'baleyesopen_score': row.get('baleyesopen_score', ''),

            ## Comments 
            'comments': row.get('Comments', row.get('comments', '')),
        }
        ctx = {k: display_value(v) for k, v in ctx.items()}

        fname_name = sanitize_fname(str(row.get('Name', f'student_{i}')))
        fname_sr = sanitize_fname(str(row.get('Sr No', i)))
        out_path = os.path.join(out_dir, f"report_{fname_sr}_{fname_name}.html")
        with open(out_path, 'w', encoding='utf-8') as out_f:
            out_f.write(template.render(**ctx))


def convert_htmls_to_pdfs(html_out_dir: str, pdf_out_dir: str, wkhtmltopdf_path: str | None = None) -> None:
    """Convert all .html files in html_out_dir to .pdf using wkhtmltopdf and write PDFs to pdf_out_dir."""
    if wkhtmltopdf_path is None:
        wkhtmltopdf_path = shutil.which('wkhtmltopdf')
    if not wkhtmltopdf_path:
        raise FileNotFoundError('wkhtmltopdf not found on PATH; please install it (apt-get install wkhtmltopdf)')

    html_files = sorted(glob.glob(os.path.join(html_out_dir, '*.html')))
    if not html_files:
        print(f'No HTML files found in {html_out_dir} to convert.')
        return

    os.makedirs(pdf_out_dir, exist_ok=True)

    for html in html_files:
        pdf_fname = os.path.splitext(os.path.basename(html))[0] + '.pdf'
        pdf_path = os.path.join(pdf_out_dir, pdf_fname)
        try:
            subprocess.run([wkhtmltopdf_path, "--enable-local-file-access", html, pdf_path],check=True)
            print(f'Converted: {html} -> {pdf_path}')
        except subprocess.CalledProcessError as e:
            print(f'Error converting {html}: {e}')

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