import os 
import jinja2
import pandas as pd 

from utils.formatting import display_value, sanitize_fname
from utils.paths import repo_root

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
    REPO_ROOT = repo_root()
    logo_path = REPO_ROOT / "assets/images/wyrz_logo.svg"
    for i, row in df_out.head(n).iterrows():
        # map many possible column names to the template variables
        ctx = {
            'logo_url': logo_path.resolve().as_uri(),
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


