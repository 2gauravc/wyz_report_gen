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

# Code Flow 

## Ingestion -> CSV Report Generation

### Expected Output 

The output creates a report with the following cols: 

dataset: Entity Cols (6)
srno,name,class,grade,section,gender,

dataset: measurement cols (14)
height_cm,weight_kg,slr_left,slr_right,smbt_cm,sprint_20m_sec,oh_mbp_60s_reps,shoulder_mobility_left,shoulder_mobility_right,sbj_cm,pro_agility_505_sec,chair_squat_60s_reps,balance_eo_left_sec,balance_eo_right_sec,

dataset: derived attributes (1)
bmi,

metrics (level0) raw:9
chair_squat_60s,oh_mbp_60s,standing_broad_jump,seated_medicine_ball_throw,sprint_20m,pro_agility_505,straight_leg_raise,shoulder_mobility,balance_eyes_open,

metrics (level0) agg: 27
chair_squat_60s_avg,chair_squat_60s_p80,chair_squat_60s_best,oh_mbp_60s_avg,oh_mbp_60s_p80,oh_mbp_60s_best,standing_broad_jump_avg,standing_broad_jump_p80,standing_broad_jump_best,seated_medicine_ball_throw_avg,seated_medicine_ball_throw_p80,seated_medicine_ball_throw_best,sprint_20m_avg,sprint_20m_p80,sprint_20m_best,pro_agility_505_avg,pro_agility_505_p80,pro_agility_505_best,straight_leg_raise_avg,straight_leg_raise_p80,straight_leg_raise_best,shoulder_mobility_avg,shoulder_mobility_p80,shoulder_mobility_best,balance_eyes_open_avg,balance_eyes_open_p80,balance_eyes_open_best

metrics (level0) score: 9
chair_squat_60s_score,oh_mbp_60s_score,standing_broad_jump_score,seated_medicine_ball_throw_score,sprint_20m_score,pro_agility_505_score,straight_leg_raise_score,shoulder_mobility_score,balance_eyes_open_score
