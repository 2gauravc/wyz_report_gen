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

```
python src/student_report.py --app-config config/app_config.yaml --run-config config/report_run_config.yaml
```

## Run the school report

```
python school_report.py --csv out_csv/report_output.csv --out out_html/school_summary.html --template school_report_template.html
```

