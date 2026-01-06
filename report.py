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
		ctx = {
			'logo_url': '',
			'school_name': row.get('SchoolName', ''),
			'name': row.get('Name', ''),
			'height': row.get('Height', ''),
			'fitness_score': row.get('FitnessScore', ''),
			'class_name': row.get('Class', ''),
			'gender': row.get('Gender', ''),
			'weight': row.get('Weight', ''),
			'recorded_date': row.get('RecordedDate', ''),

			# Chair squat specific values
			'chair_measure': row.get('ChairSquats1min', ''),
			'chair_class_avg': row.get('ChairSquats1min_avg', ''),
			'chair_80th': row.get('ChairSquats1min80th percentile', ''),
			'chair_best': row.get('ChairSquats1min_best', ''),
			'chair_score': int(row['Score']) if pd.notna(row.get('Score')) else '',

			# Medicine ball press values (if detected)
			'mbp_measure': row.get('MBP_Measure', ''),
			'mbp_class_avg': row.get('MBP_avg', ''),
			'mbp_80th': row.get('MBP_80th percentile', ''),
			'mbp_best': row.get('MBP_best', ''),
			'mbp_score': int(row['MBP_Score']) if pd.notna(row.get('MBP_Score')) else '',
		}

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
            subprocess.run([wkhtmltopdf_path, html, pdf_path], check=True)
            print(f'Converted: {html} -> {pdf_path}')
        except subprocess.CalledProcessError as e:
            print(f'Error converting {html}: {e}')

def process(input_csv: str, output_csv: str) -> pd.DataFrame:
	df = pd.read_csv(input_csv)

	# Identify columns flexibly
	sr_col = find_column(df.columns, ["srno", "sr", "sno", "srno"])
	name_col = find_column(df.columns, ["name"]) or find_column(df.columns, ["student"]) or None
	class_col = find_column(df.columns, ["class", "grade"])
	gender_col = find_column(df.columns, ["bg", "boy", "girl", "gender"])
	chair_col = find_column(df.columns, ["chairsquat", "chairsquats", "chairsquats1min", "chairsquat1min", "chairsquat1", "chairsquat1min"]) or find_column(df.columns, ["chair", "squat"]) or None

	if chair_col is None:
		# try any column containing 'chair' or 'squat' words
		chair_col = find_column(df.columns, ["chair", "squat"]) if find_column(df.columns, ["chair", "squat"]) else None

	if name_col is None:
		# fallback to first non-numeric column
		for col in df.columns:
			if df[col].dtype == object:
				name_col = col
				break

	# Detect additional columns: school, height, weight, fitness score, recorded date, medicine ball press
	school_col = find_column(df.columns, ["school", "schoolname"]) or None
	height_col = find_column(df.columns, ["height", "ht"]) or None
	weight_col = find_column(df.columns, ["weight", "wt"]) or None
	fitness_col = find_column(df.columns, ["fitness", "fitnessscore", "score"]) or None
	date_col = find_column(df.columns, ["date", "recorded", "recordeddate", "dataasrecorded"]) or None
	mbp_col = find_column(df.columns, ["medicineball", "medicine", "ballpress", "mbp", "medicineballpress"]) or None

	# Build working DataFrame with standard column names
	work = pd.DataFrame()
	work["Sr No"] = df[sr_col] if sr_col in df.columns else pd.RangeIndex(start=1, stop=len(df) + 1)
	work["Name"] = df[name_col] if name_col in df.columns else ""
	work["Class"] = df[class_col] if class_col in df.columns else ""
	work["Gender"] = df[gender_col] if gender_col in df.columns else ""
	if chair_col and chair_col in df.columns:
		work["ChairSquats1min"] = pd.to_numeric(df[chair_col], errors="coerce")
	else:
		work["ChairSquats1min"] = pd.to_numeric(pd.Series([np.nan] * len(df)), errors="coerce")

	# optional metadata columns
	work['SchoolName'] = df[school_col] if school_col in df.columns else ''
	work['Height'] = df[height_col] if height_col in df.columns else ''
	work['Weight'] = df[weight_col] if weight_col in df.columns else ''
	work['FitnessScore'] = df[fitness_col] if fitness_col in df.columns else ''
	work['RecordedDate'] = df[date_col] if date_col in df.columns else ''

	# Medicine Ball Press detection
	if mbp_col and mbp_col in df.columns:
		work['MBP_Measure'] = pd.to_numeric(df[mbp_col], errors='coerce')
	else:
		work['MBP_Measure'] = pd.to_numeric(pd.Series([np.nan] * len(df)), errors='coerce')

	# Compute per (Class, Gender) statistics
	grp = work.groupby(["Class", "Gender"])['ChairSquats1min']
	work['ChairSquats1min_avg'] = grp.transform('mean')
	work['ChairSquats1min80th percentile'] = grp.transform(lambda x: np.nanpercentile(x.dropna(), 80) if x.dropna().size > 0 else np.nan)
	work['ChairSquats1min_best'] = grp.transform('max')

	# Score in 1..5 by dividing group's range into 5 equal buckets
	work['Score'] = grp.transform(lambda x: assign_scores(x, bins_count=5))

	# Compute MBP stats per same grouping
	mbp_grp = work.groupby(["Class", "Gender"])['MBP_Measure']
	work['MBP_avg'] = mbp_grp.transform('mean')
	work['MBP_80th percentile'] = mbp_grp.transform(lambda x: np.nanpercentile(x.dropna(), 80) if x.dropna().size > 0 else np.nan)
	work['MBP_best'] = mbp_grp.transform('max')
	work['MBP_Score'] = mbp_grp.transform(lambda x: assign_scores(x, bins_count=5))

	# Reorder columns as requested
	out_cols = [
		"Sr No", "Name", "Class", "Gender",
		"ChairSquats1min", "ChairSquats1min_avg", "ChairSquats1min80th percentile", "ChairSquats1min_best", "Score",
		"MBP_Measure", "MBP_avg", "MBP_80th percentile", "MBP_best", "MBP_Score",
		"SchoolName", "Height", "Weight", "FitnessScore", "RecordedDate",
	]
	out = work[out_cols]

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
			render_html_for_first_n(df_out, args.template, args.html_outdir, 10)
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

