# Install and Run the Code 

## Install wkhtmltopdf
```bash
sudo apt-get update 
sudo apt-get install -y wkhtmltopdf
wkhtmltopdf --version
```

## Install packages 

```
pip install -r requirements.txt
```


## Run the individual student report  

Render HTML and PDF 
```
python3 src/student_report.py data/input_data/master_6th_class.csv -o output/out_csv/report_output.csv --render-html --template templates/student_report_template.html --html-outdir output/out_html --pdf-outdir output/out_pdf --limit 2 --to-pdf
```

Only generate CSV. do not render HTMl and PDF 
```
python3 src/student_report.py data/input_data/master_6th_class.csv \
-o output/out_csv/report_output.csv --config config/metrics/fitness_metrics.yaml

```


## Run the school report

```
python school_report.py --csv out_csv/report_output.csv --out out_html/school_summary.html --template school_report_template.html
```

