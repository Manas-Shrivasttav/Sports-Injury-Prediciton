# Data — Sports Injury Prediction Project

## Overview

The dataset used in this project is **synthetic** — generated using [Gretel.ai's](https://gretel.ai) ACTGAN model — due to the limited availability, ethical constraints, and privacy concerns around real-world sports injury data.

## Dataset Summary

| Attribute | Detail |
|---|---|
| File name | `synthetic_data.csv` |
| Total records | 5,000 |
| Features | 17 (16 independent + 1 target) |
| Target classes | 4 injury types |
| Generation tool | [Gretel.ai](https://gretel.ai) — ACTGAN |

## Class Distribution

| Injury Type | Records |
|---|---|
| Hamstring | 1,443 |
| ACL Tear | 1,392 |
| Sprained Ankle | 1,363 |
| Lower Back Pain | 802 |
| **Total** | **5,000** |

## How the Data Was Generated

1. A sample dataset was manually created with real-world-informed values across all 17 variables and all 4 injury types
2. This sample was fed into **Gretel.ai** using the **ACTGAN** (Adversarial Conditional GAN) architecture
3. Gretel.ai augmented the sample into 5,000 synthetic records that mirror real injury distribution patterns
4. The synthetic dataset was validated for plausibility against published sports science literature

## Generating Your Own Dataset

To reproduce or extend the dataset:

1. Create a sample CSV (minimum 50 rows per class) using the 17 variables below
2. Sign up at [gretel.ai](https://gretel.ai) — free tier available
3. Create a new project → select **ACTGAN** model
4. Upload your sample CSV as training data
5. Set output record count to 5,000
6. Download the generated dataset
7. Save as `synthetic_data.csv` in the `data/` folder

## Variable Descriptions

| Variable | Type | Description |
|---|---|---|
| Injury Type | Categorical (target) | ACL Tear / Hamstring / Lower Back Pain / Sprained Ankle |
| Recovery Time | Numerical | Weeks to recover from previous injury |
| Medical History | Categorical | Latest previous injury type |
| Player Age | Numerical | Age in years |
| Player Height | Numerical | Height in cm |
| Player Weight | Numerical | Weight in kg |
| Frequency of Workouts | Numerical | Training sessions per week |
| Intensity of Workouts | Categorical | Low / Medium / High |
| Specialized Training Programs | Categorical | Training program type followed |
| Game Weather | Categorical | Weather condition during match |
| Game Field Condition | Categorical | Field surface condition |
| Yellow Cards | Numerical | Yellow cards received in match |
| Red Cards | Numerical | Red cards received in match |
| Minutes Played | Numerical | Average minutes per match |
| Average Distance Covered | Numerical | Average km covered per match |
| Playing Style | Categorical | Team's tactical playing style |
| Coaching Staff | Categorical | Level of coaching staff expertise |

## Categorical Variables Encoding

The following columns are dummy-encoded during preprocessing:

`Medical History` · `Intensity of Workouts` · `Specialized Training Programs` · `Game Weather` · `Game Field Condition` · `Playing Style` · `Coaching Staff`

## Note on Data Limitations

- The dataset is synthetic and may not capture all real-world nuances of football injury patterns
- The Sprained Ankle class showed the weakest model performance (AUC: 0.63), which may reflect limitations in synthetic generation for this class
- Real-world validation with club-sourced data is recommended before deploying any model in a professional sports setting
