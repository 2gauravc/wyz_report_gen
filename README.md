# Install and Run the Code 

## Install wkhtmltopdf
```bash
sudo apt-get update && sudo apt-get install -y wkhtmltopdf
```

## Run the code 
```
python3 report.py input_data/master_6th_class.csv -o out_csv/report_output.csv --render-html --template report_template.html --html-outdir out_html --pdf-outdir out_pdf --limit 10 --to-pdf
```

# Business Logic 

## Step 1: Compute the cohort averages 

Metrics: 
1. MaximumChairSquats60seconds - (Count)
2. MaximumMedicineBallPress60seconds - (Count)
3. StandingBroadJumpDistance - (Distance in m)
4. SeatedMedicineBallThrowDistance(2kg) - (Distance in m)
5. 20mRunningTime(20m) - (Time is seconds)
6. ProAgilityTime - (Time in seconds)
7. ActiveStraightLegRaise(SLR) - (?)
8. ShoulderMobility - (?)
9. BalanceOneLegTimeLeft_EyesOpen - (Time in seconds, max 30)
10. BalanceOneLegTimeRight_EyesOpen - (Time in seconds max 30)
11. BalanceOneLegTimeLeft_EyesClosed - (Time in seconds max 30)
12. BalanceOneLegTimeRight_EyesClosed - (Time in seconds max 30)

For each metric: 
Average - By Gender 
80th Percentile - By Gender 
Best - By Gender 
20th, 40th, 60th, 80th percentile - By Gender 