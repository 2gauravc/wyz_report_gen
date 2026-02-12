# src/render/html.py
from __future__ import annotations

import os
from datetime import date
from pathlib import Path
from typing import Optional

import jinja2
import pandas as pd

# assuming you already have these helpers somewhere (as in your current codebase)
from utils.paths import repo_root
from utils.formatting import display_value, sanitize_fname
from utils.yamlops import load_yaml_config

def _as_file_uri(path: Path) -> str:
    try:
        return path.resolve().as_uri()
    except Exception:
        return str(path)


def _resolve_logo_urls(cfg: dict) -> dict:
    """
    Resolves platform + school logos based on cfg.report_context.logos.
    Returns dict: {logo_platform_url, logo_school_url}
    """
    rc = (cfg.get("report_context") or {})
    logos = (rc.get("logos") or {})

    REPO_ROOT = repo_root()

    platform = logos.get("platform") or {}
    school = logos.get("school") or {}

    # platform logo is a repo_asset with path
    platform_path = platform.get("path")
    logo_platform_url = ""
    if platform_path:
        logo_platform_url = _as_file_uri((REPO_ROOT / platform_path))

    # school logo optional (often passed via CLI/config) — here we just support config default if present
    # If you later pass a CLI arg for school logo, set cfg["report_context"]["logos"]["school"]["default"] to that path
    school_default = school.get("default")
    logo_school_url = ""
    if school_default:
        p = Path(school_default)
        if not p.is_absolute():
            p = REPO_ROOT / p
        if p.exists():
            logo_school_url = _as_file_uri(p)

    return {
        "logo_platform_url": logo_platform_url,
        "logo_school_url": logo_school_url,
    }


def _report_texts(cfg: dict) -> dict:
    rc = (cfg.get("report_context") or {})
    school = (rc.get("school_name") or {})
    title = (rc.get("report_title") or {})

    return {
        "school_name": school.get("default", "WYRDZ School"),
        "report_title": title.get("default", "Individual Fitness Testing & Assessment"),
    }


def _build_student_ctx(row: pd.Series, cfg: dict) -> dict:
    """
    Build a single context dict for the 2-page template.
    Uses your *canonical* output column names (level0/1/2/3 + aggregates).
    """
    texts = _report_texts(cfg)
    logos = _resolve_logo_urls(cfg)

    # date: prefer row['date'] if you ever add it; else today
    eval_date = row.get("date")
    if pd.isna(eval_date) or eval_date is None or str(eval_date).strip() == "":
        eval_date = date.today().strftime("%d.%m.%Y")

    ctx = {
        # report context
        **texts,
        **logos,
        "date": eval_date,

        # about student
        "srno": row.get("srno", ""),
        "name": row.get("name", ""),
        "class": row.get("class", ""),
        "grade": row.get("grade", ""),
        "section": row.get("section", ""),
        "gender": row.get("gender", ""),

        # basic measurements
        "height_cm": row.get("height_cm", ""),
        "weight_kg": row.get("weight_kg", ""),
        "bmi": row.get("bmi", ""),
        "fitnessscore": row.get("fitnessscore", ""),
        "maxfitnessscore": row.get("maxfitnessscore", ""),

        # level2 grid
        "foundation_score": row.get("foundation_score", ""),
        "movement_skills_score": row.get("movement_skills_score", ""),
        "level2_grid_2x2": row.get("level2_grid_2x2", ""),

        # level1 domain snapshot (scores + aggregates)
        "strength_score": row.get("strength_score", ""),
        "strength_avg": row.get("strength_avg", ""),
        "strength_p80": row.get("strength_p80", ""),

        "power_score": row.get("power_score", ""),
        "power_avg": row.get("power_avg", ""),
        "power_p80": row.get("power_p80", ""),

        "speed_agility_score": row.get("speed_agility_score", ""),
        "speed_agility_avg": row.get("speed_agility_avg", ""),
        "speed_agility_p80": row.get("speed_agility_p80", ""),

        "flexibility_score": row.get("flexibility_score", ""),
        "flexibility_avg": row.get("flexibility_avg", ""),
        "flexibility_p80": row.get("flexibility_p80", ""),

        "balance_score": row.get("balance_score", ""),
        "balance_avg": row.get("balance_avg", ""),
        "balance_p80": row.get("balance_p80", ""),

        # page 2: level0 tables (value + aggregates + best + score)
        "chair_squat_60s": row.get("chair_squat_60s", ""),
        "chair_squat_60s_avg": row.get("chair_squat_60s_avg", ""),
        "chair_squat_60s_p80": row.get("chair_squat_60s_p80", ""),
        "chair_squat_60s_best": row.get("chair_squat_60s_best", ""),
        "chair_squat_60s_score": row.get("chair_squat_60s_score", ""),

        "oh_mbp_60s": row.get("oh_mbp_60s", ""),
        "oh_mbp_60s_avg": row.get("oh_mbp_60s_avg", ""),
        "oh_mbp_60s_p80": row.get("oh_mbp_60s_p80", ""),
        "oh_mbp_60s_best": row.get("oh_mbp_60s_best", ""),
        "oh_mbp_60s_score": row.get("oh_mbp_60s_score", ""),

        "standing_broad_jump": row.get("standing_broad_jump", ""),
        "standing_broad_jump_avg": row.get("standing_broad_jump_avg", ""),
        "standing_broad_jump_p80": row.get("standing_broad_jump_p80", ""),
        "standing_broad_jump_best": row.get("standing_broad_jump_best", ""),
        "standing_broad_jump_score": row.get("standing_broad_jump_score", ""),

        "seated_medicine_ball_throw": row.get("seated_medicine_ball_throw", ""),
        "seated_medicine_ball_throw_avg": row.get("seated_medicine_ball_throw_avg", ""),
        "seated_medicine_ball_throw_p80": row.get("seated_medicine_ball_throw_p80", ""),
        "seated_medicine_ball_throw_best": row.get("seated_medicine_ball_throw_best", ""),
        "seated_medicine_ball_throw_score": row.get("seated_medicine_ball_throw_score", ""),

        "sprint_20m": row.get("sprint_20m", ""),
        "sprint_20m_avg": row.get("sprint_20m_avg", ""),
        "sprint_20m_p80": row.get("sprint_20m_p80", ""),
        "sprint_20m_best": row.get("sprint_20m_best", ""),
        "sprint_20m_score": row.get("sprint_20m_score", ""),

        "pro_agility_505": row.get("pro_agility_505", ""),
        "pro_agility_505_avg": row.get("pro_agility_505_avg", ""),
        "pro_agility_505_p80": row.get("pro_agility_505_p80", ""),
        "pro_agility_505_best": row.get("pro_agility_505_best", ""),
        "pro_agility_505_score": row.get("pro_agility_505_score", ""),

        "straight_leg_raise": row.get("straight_leg_raise", ""),
        "straight_leg_raise_avg": row.get("straight_leg_raise_avg", ""),
        "straight_leg_raise_p80": row.get("straight_leg_raise_p80", ""),
        "straight_leg_raise_best": row.get("straight_leg_raise_best", ""),
        "straight_leg_raise_score": row.get("straight_leg_raise_score", ""),

        "shoulder_mobility": row.get("shoulder_mobility", ""),
        "shoulder_mobility_avg": row.get("shoulder_mobility_avg", ""),
        "shoulder_mobility_p80": row.get("shoulder_mobility_p80", ""),
        "shoulder_mobility_best": row.get("shoulder_mobility_best", ""),
        "shoulder_mobility_score": row.get("shoulder_mobility_score", ""),

        "balance_eyes_open": row.get("balance_eyes_open", ""),
        "balance_eyes_open_avg": row.get("balance_eyes_open_avg", ""),
        "balance_eyes_open_p80": row.get("balance_eyes_open_p80", ""),
        "balance_eyes_open_best": row.get("balance_eyes_open_best", ""),
        "balance_eyes_open_score": row.get("balance_eyes_open_score", ""),
    }

    # final formatting pass (your existing helper)
    ctx = {k: display_value(v) for k, v in ctx.items()}
    return ctx


def render_html_for_first_n(
    df_out: pd.DataFrame,
    template_path: str,
    out_dir: str,
    config_path: str,
    n: int = 10
) -> None:
    """
    Renders 2-page student reports using a single template with a page break.
    """
    cfg = load_yaml_config(config_path)
   
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Template not found: {template_path}")

    os.makedirs(out_dir, exist_ok=True)

    with open(template_path, "r", encoding="utf-8") as f:
        tmpl_src = f.read()

    env = jinja2.Environment(autoescape=False)
    template = env.from_string(tmpl_src)

    for i, row in df_out.head(n).iterrows():
        ctx = _build_student_ctx(row, cfg)

        fname_name = sanitize_fname(str(row.get("name", f"student_{i}")))
        fname_sr = sanitize_fname(str(row.get("srno", i)))
        out_path = os.path.join(out_dir, f"report_{fname_sr}_{fname_name}.html")

        with open(out_path, "w", encoding="utf-8") as out_f:
            out_f.write(template.render(**ctx))
