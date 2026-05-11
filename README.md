# Early Detection of Sports Injury in Professional Football Players

[![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat&logo=python)](https://python.org)
[![Framework](https://img.shields.io/badge/Framework-Scikit--learn%20%7C%20Keras-orange?style=flat)]()
[![Type](https://img.shields.io/badge/Task-Multi--Class%20Classification-blue?style=flat)]()
[![Best Model](https://img.shields.io/badge/Best%20Model-AdaBoost%20%7C%20Acc%3A%200.57-green?style=flat)]()
[![Data](https://img.shields.io/badge/Data-Synthetic%20(Gretel.ai)-purple?style=flat)]()

---

## 🧠 Project Overview

Football players are among the highest-risk athletes for injury — ACL tears, hamstring strains, lower back pain, and sprained ankles alone account for a significant proportion of professional career disruptions. Yet most clubs still rely on reactive treatment rather than proactive prediction.

This project builds and compares **four machine learning and deep learning models** to predict which type of injury a professional football player is likely to sustain, using 17 player-specific variables including training load, physical metrics, game conditions, and medical history.

The best-performing model — **AdaBoost** — provides coaches and medical staff with actionable, early-warning injury predictions, enabling personalised prevention strategies before injuries occur.

---

## 🎯 Objectives

- Build multi-class classification models to predict 4 injury types in football players
- Compare AdaBoost, Random Forest, XGBoost, and Deep Neural Network across precision, recall, F1-score, and ROC-AUC
- Identify the best-performing model and translate predictions into **concrete prevention strategies** for coaches and managers

---

## 📊 Dataset

| Attribute | Detail |
|---|---|
| Total records | 5,000 (synthetic) |
| Generation method | Sample data manually crafted → augmented via **Gretel.ai (ACTGAN)** |
| Target variable | Injury Type (4 classes) |
| Features | 17 variables |
| Class distribution | ACL Tear: 1,392 · Hamstring: 1,443 · Sprained Ankle: 1,363 · Lower Back Pain: 802 |

### Why Synthetic Data?
Real-world football injury data is scarce, private, and ethically constrained. Synthetic data generated via ACTGAN (Adversarial Conditional Generative Adversarial Network) mimics the statistical properties of real injury data while preserving privacy — enabling a sufficiently large dataset for model training.

### Feature Variables

| Variable | Type | Description |
|---|---|---|
| Injury Type | Categorical (target) | ACL Tear, Hamstring, Lower Back Pain, Sprained Ankle |
| Recovery Time | Numerical | Recovery time from previous injury (weeks) |
| Medical History | Categorical | Latest previous injury type |
| Player Age | Numerical | In years |
| Player Height | Numerical | In cm |
| Player Weight | Numerical | In kg |
| Frequency of Workouts | Numerical | Sessions per week |
| Intensity of Workouts | Categorical | Low / Medium / High |
| Specialized Training Programs | Categorical | Training program type |
| Game Weather | Categorical | Weather during injury match |
| Game Field Condition | Categorical | Field condition during injury match |
| Yellow Cards | Numerical | Cards received in match |
| Red Cards | Numerical | Cards received in match |
| Minutes Played | Numerical | Average minutes per match |
| Average Distance Covered | Numerical | km per match |
| Playing Style | Categorical | Team's playing style |
| Coaching Staff | Categorical | Staff expertise level |

---

## 🏗️ Methodology

```
Raw Sample Data (17 variables)
        │
        ▼
  Synthetic Data Generation (Gretel.ai ACTGAN)
  → 5,000 records across 4 injury classes
        │
        ▼
  Data Pre-processing
  ├── Dummy encoding (categorical → numerical)
  ├── Label encoding (target variable)
  └── StandardScaler (feature scaling)
        │
        ▼
  ┌──────────────────────────────────────────────┐
  │     Model Training + GridSearchCV (5-fold)   │
  │                                              │
  │  ┌─────────────┐  ┌─────────────┐            │
  │  │  AdaBoost   │  │   XGBoost   │            │
  │  │  ← BEST     │  │             │            │
  │  └─────────────┘  └─────────────┘            │
  │  ┌─────────────┐  ┌─────────────┐            │
  │  │   Random    │  │    DNN      │            │
  │  │   Forest    │  │  (Keras)    │            │
  │  └─────────────┘  └─────────────┘            │
  └──────────────────────────────────────────────┘
        │
        ▼
  Evaluation: Precision · Recall · F1 · ROC-AUC
        │
        ▼
  Injury-specific Prevention Recommendations
```

---

## 📈 Results

### Model Comparison

| Model | Precision (Wtd.) | Recall (Wtd.) | F1-Score (Wtd.) | Accuracy |
|---|---|---|---|---|
| **AdaBoost** | **0.55** | **0.57** | **0.56** | **0.57** |
| XGBoost | 0.54 | 0.56 | 0.55 | 0.56 |
| Random Forest | 0.54 | 0.56 | 0.54 | 0.56 |
| DNN (Keras) | 0.53 | 0.55 | 0.53 | 0.55 |

**AdaBoost outperformed all models** across every metric and was selected as the primary model.

### AdaBoost — Per-Class Results

| Injury Type | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|
| ACL Tear | 0.65 | 0.72 | 0.68 | **0.85** |
| Hamstring | 0.59 | 0.73 | 0.66 | 0.84 |
| Lower Back Pain | 0.51 | 0.45 | 0.48 | 0.83 |
| Sprained Ankle | 0.45 | 0.32 | 0.37 | 0.63 |

### Optimised Hyperparameters

| Model | Best Parameters |
|---|---|
| AdaBoost | `n_estimators=200`, `learning_rate=0.1` |
| Random Forest | `max_depth=10`, `max_features=2`, `min_samples_split=4`, `n_estimators=200` |
| XGBoost | `learning_rate=0.1`, `max_depth=3`, `n_estimators=100` |
| DNN | `neurons=128`, `activation=softmax`, `optimizer=Adamax`, `lr=0.01`, `dropout=0`, `weight_constraint=4`, `epochs=40`, `batch_size=128` |

---

## 🛡️ Injury Prevention Recommendations

Based on AdaBoost predictions, the following actionable strategies are recommended for coaches and managers:

| Injury | Workout Frequency | Workout Intensity | Specialised Training | Avg. Min/Match | Avg. Distance |
|---|---|---|---|---|---|
| ACL Tear | 4–5×/week | Moderate–High | Neuromuscular control + plyometrics | 60–90 min | 9–11 km |
| Hamstring | 4–5×/week | Moderate–High + warm-up | Eccentric strengthening | 60–90 min | 9–11 km |
| Lower Back | 3–4×/week | Low–Moderate + core | Yoga / Pilates | 45–60 min | 7–9 km |
| Sprained Ankle | 3–4×/week | Low–Moderate + balance | Ankle stability + agility | 45–60 min | 7–9 km |

---

## 📁 Repository Structure

```
├── notebooks/
│   ├── adaboost_model.py            # Best model — AdaBoost ← start here
│   ├── random_forest_model.py       # Random Forest classifier
│   ├── xgboost_model.py             # XGBoost classifier
│   └── neural_network_model.py      # Deep Neural Network (Keras)
├── data/
│   └── DATA_README.md               # Dataset description & generation guide
└── README.md
```

---

## 🚀 How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
tensorflow
scikeras
```

### Run Any Model

```bash
# Clone the repo
git clone https://github.com/Manas-Shrivasttav/Sports-Injury-Prediction-Football
cd Sports-Injury-Prediction-Football

# Install dependencies
pip install -r requirements.txt

# Add your data file (see data/DATA_README.md for how to generate it)
# Place it at: data/synthetic_data.csv

# Run the best performing model
python notebooks/adaboost_model.py

# Or run any other model
python notebooks/random_forest_model.py
python notebooks/xgboost_model.py
python notebooks/neural_network_model.py
```

Each script runs end-to-end: loads data → preprocesses → tunes hyperparameters → trains → evaluates → saves plots to `plots/`.

### Generate Synthetic Data

The original dataset was generated using [Gretel.ai](https://gretel.ai). See `data/DATA_README.md` for a step-by-step guide to reproduce it.

---

## 🛠️ Tech Stack

`Python` `Scikit-learn` `XGBoost` `TensorFlow` `Keras` `scikeras` `Pandas` `NumPy` `Matplotlib` `Seaborn` `Gretel.ai`

---

## 🔮 Future Work

- Collect real-world injury data from football clubs to validate findings
- Extend to other sports (basketball, cricket, rugby)
- Add biomechanical and GPS tracking data as additional features
- Build a real-time injury risk dashboard for coaching staff
- Address Sprained Ankle class imbalance (lowest AUC: 0.63) using SMOTE

---

## 👤 Author

**Manas Shrivastav**

---

## 📄 License

Open for academic and research use.
