# Install and Run the Code 

## Install wkhtmltopdf
```bash
sudo apt-get update && sudo apt-get install -y wkhtmltopdf
```

## Run the individual student report  

Render HTML and PDF 
```
python3 src/student_report.py data/input_data/master_6th_class.csv -o output/out_csv/report_output.csv --render-html --template templates/student_report_template.html --html-outdir output/out_html --pdf-outdir output/out_pdf --limit 2 --to-pdf
```

Only generate CSV. do not render HTMl and PDF 
```
python3 student_report.py input_data/master_6th_class.csv \
  -o out_csv/report_output.csv

```


## Run the school report

```
python school_report.py --csv out_csv/report_output.csv --out out_html/school_summary.html --template school_report_template.html
```

# Observed Metric and Reporting Logic 

## Level 0 Metrics 

| **Sr. No.** | **Report Metric**               | **Short Name** | **How It’s Measured**                           | **Grouping Dimensions** | **Aggregate Stats**            | **Scoring Method**                                             |
| ----------: | ------------------------------- | -------------- | ----------------------------------------------- | ----------------------- | ------------------------------ | -------------------------------------------------------------- |
|       **1** | Maximum Chair Squats            | `chairsquat`   | Count of squats completed in **60 seconds**     | Gender, Class (Grade)   | Average, 80th percentile, Best | **Score (1–5)** using min–max range bins                       |
|       **2** | Maximum Medicine Ball Press     | `ohmbp`        | Count of presses completed in **60 seconds**    | Gender, Class (Grade)   | Average, 80th percentile, Best | **Score (1–5)** using min–max range bins                       |
|       **3** | Standing Broad Jump             | `sbj`          | Jump distance measured in **meters**            | Gender, Class (Grade)   | Average, 80th percentile, Best | **Score (1–5)** using min–max range bins                       |
|       **4** | Seated Medicine Ball Throw      | `smbt`         | Throw distance in **meters** (2 kg ball)        | Gender, Class (Grade)   | Average, 80th percentile, Best | **Score (1–5)** using min–max range bins                       |
|       **5** | 20 m Run Speed                  | `runspeed`     | Time taken to cover **20 meters (seconds)**     | Gender, Class (Grade)   | Average, 80th percentile, Best | **Score (1–5)** using min–max range bins *(lower is better)*   |
|       **6** | Pro-Agility (5-0-5)             | `proagility`   | Time taken in **seconds**                       | Gender, Class (Grade)   | Average, 80th percentile, Best | **Score (1–5)** using min–max range bins *(lower is better)*   |
|       **7** | Active Straight Leg Raise (FMS) | `slr`          | Left & right assessed using **FMS scale (1–3)** | Gender, Class (Grade)   | Average, Best                  | **Ordinal mapping:** 1→1, 2→3, 3→5 (worse of L/R)              |
|       **8** | Shoulder Mobility (FMS)         | `shouldermob`  | Left & right assessed using **FMS scale (1–3)** | Gender, Class (Grade)   | Average, Best                  | **Ordinal mapping:** 1→1, 2→3, 3→5 (worse of L/R)              |
|       **9** | Balance – Eyes Open             | `baleyesopen`  | Single-leg balance hold time in **seconds**     | Gender, Class (Grade)                     | Average, Best                           | **Fixed scale:** 1–6s→1, 7–12s→2, 13–18s→3, 19–24s→4, 25–30s→5 |

## Level 1 Metrics 