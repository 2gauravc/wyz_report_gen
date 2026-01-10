# Install and Run the Code 

## Install wkhtmltopdf
```bash
sudo apt-get update && sudo apt-get install -y wkhtmltopdf
```

## Run the code 
```
python3 report.py input_data/master_6th_class.csv -o out_csv/report_output.csv --render-html --template report_template.html --html-outdir out_html --pdf-outdir out_pdf --limit 10 --to-pdf
```

# Observed Metric and Reporting Logic 

| Category        | Report Metric               | Short Name | Observation Specifics                                                   | Dimensions             | Derivation                                                         |
|-----------------|-----------------------------|------------|-------------------------------------------------------------------------|------------------------|--------------------------------------------------------------------|
| Strength        | Maximum Chair Squats        |  chairsquat          | Observation: Count Time Limit: 60s                                      | Gender,  Class (Grade) | Avg, 80th Percentile, Best Score (1-5) based on min-max range bins |
|                 | Maximum Medicine Ball Press | OHMBP      | Observation: Count Time Limit: 60s                                      | Gender,  Class (Grade) | Avg, 80th Percentile, Best Score (1-5) based on min-max range bins |
| Power           | Standard Broad Jump         | SBJ        | Observation: Distance (m)                                               | Gender,  Class (Grade) | Avg, 80th Percentile, Best Score (1-5) based on min-max range bins |
|                 | Seated Medicine Ball Throw  | SMBT       | Observation: Distance (m) Ball Weight : 2 kgs                           | Gender,  Class (Grade) | Avg, 80th Percentile, Best Score (1-5) based on min-max range bins |
| Speed & Agility | Max Running Speed           |  runspeed   | Observation: Time (s) Distance: 20m                                     | Gender,  Class (Grade) | Avg, 80th Percentile, Best Score (1-5) based on min-max range bins |
|                 | Pro Agility                 |proagility    | Observation: Time (s)                                                   | Gender,  Class (Grade) | Avg, 80th Percentile, Best Score (1-5) based on min-max range bins |
| Flexibility     | Active Straight Leg Raise)  | SLR        | Measure both Left and Right  Grade according to FMS grading scale (1-3) | Gender,  Class (Grade) | Measure: Worse (lower) of Left, Right  Avg, Best                   |
|                 | Shoulder Mobility           |   shouldermob         | Measure both Left and Right  Grade according to FMS grading scale (1-3) | Gender,  Class (Grade) | Measure: Worse (lower) of Left, Right Avg, Best                    |
| Balance         | Balance - Eyes Open       | baleyesopen         | Observation: Time (s)                                                   | NA                     | Measure: Worse (lower) of Left, Right  Fixed Scale: 1-6s 1, 25-30s - 5                   |
