

# 🧱 LEVEL 1 — Domains (5 Metrics)

Each Level1 metric is computed as:

> **Mean of Level0 scores of its child metrics**
> Rounded to 1 decimal → then clamped to range [1–5] → then rounded to integer for `_score`

---

## 1️⃣ Strength

**Children:**

* chair_squat_60s_score
* oh_mbp_60s_score

**Formula:**

[
strength = mean(chair_squat_60s_score,; oh_mbp_60s_score)
]

[
strength_score = round_clamp(strength,; 1,; 5)
]

---

## 2️⃣ Power

**Children:**

* standing_broad_jump_score
* seated_medicine_ball_throw_score

[
power = mean(standing_broad_jump_score,; seated_medicine_ball_throw_score)
]

[
power_score = round_clamp(power,; 1,; 5)
]

---

## 3️⃣ Speed & Agility

**Children:**

* sprint_20m_score
* pro_agility_505_score

[
speed_agility = mean(sprint_20m_score,; pro_agility_505_score)
]

[
speed_agility_score = round_clamp(speed_agility,; 1,; 5)
]

---

## 4️⃣ Flexibility

**Children:**

* straight_leg_raise_score
* shoulder_mobility_score

[
flexibility = mean(straight_leg_raise_score,; shoulder_mobility_score)
]

[
flexibility_score = round_clamp(flexibility,; 1,; 5)
]

---

## 5️⃣ Balance

**Children:**

* balance_eyes_open_score

[
balance = mean(balance_eyes_open_score)
]

[
balance_score = round_clamp(balance,; 1,; 5)
]

---

# 🏗 LEVEL 2 — Higher Constructs (2 Metrics)

Level2 metrics are computed from **Level1 domain scores**.

---

## 6️⃣ Foundation

**Children:**

* strength_score
* flexibility_score
* balance_score

[
foundation = mean(strength_score,; flexibility_score,; balance_score)
]

[
foundation_score = round_clamp(foundation,; 1,; 5)
]

---

## 7️⃣ Movement Skills

**Children:**

* power_score
* speed_agility_score

[
movement_skills = mean(power_score,; speed_agility_score)
]

[
movement_skills_score = round_clamp(movement_skills,; 1,; 5)
]

---

# 🧭 Level2 View — 2×2 Grid Classification

Classification uses:

* X-axis → `foundation_score`
* Y-axis → `movement_skills_score`
* Threshold → High if ≥ 4

| Condition                       | Quadrant                      |
| ------------------------------- | ----------------------------- |
| foundation < 4 AND movement < 4 | foundation_low_movement_low   |
| foundation ≥ 4 AND movement < 4 | foundation_high_movement_low  |
| foundation < 4 AND movement ≥ 4 | foundation_low_movement_high  |
| foundation ≥ 4 AND movement ≥ 4 | foundation_high_movement_high |

---

# 🏆 LEVEL 3 — Overall Fitness (1 Metric)

Computed from **Level1 domain scores** (NOT Level2)

**Children:**

* strength_score
* power_score
* speed_agility_score
* flexibility_score
* balance_score

[
overall = mean(strength_score,; power_score,; speed_agility_score,; flexibility_score,; balance_score)
]

[
overall_score = round_clamp(overall,; 1,; 5)
]

---

# 🎯 Total Fitness Score (Absolute Score)

This is different from overall_score.

It is the **sum of Level0 scores**.

There are 9 Level0 metrics:

[
fitnessscore =
\sum_{i=1}^{9} level0_metric_score_i
]

[
maxfitnessscore = 9 \times 5 = 45
]

So:

* `fitnessscore` ranges: 9–45
* `overall_score` ranges: 1–5

---

# 📐 Summary of the Math Hierarchy

```
Level0 raw measurement
      ↓
Level0 score (1–5)
      ↓
Level1 domain = mean(Level0 scores)
      ↓
Level2 construct = mean(Level1 scores)
      ↓
Level3 overall = mean(Level1 scores)
      ↓
Total Fitness Score = sum(Level0 scores)
```

---

This now gives your README:

* Clear mathematical traceability
* Clear hierarchy
* Clear distinction between:

  * Absolute score (45-point scale)
  * Relative construct score (5-point scale)
  * Quadrant classification logic

---

