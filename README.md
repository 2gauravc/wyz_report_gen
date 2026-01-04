### Install wkhtmltopdf
```bash
sudo apt-get update && sudo apt-get install -y wkhtmltopdf
```

## Run the code 

```
python3 report.py "input_data/master_6th_class.csv" -o report_out.csv --render-html --template report_template.html --outdir out_html --limit 10 --to-pdf
````