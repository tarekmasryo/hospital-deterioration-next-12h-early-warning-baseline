# 🏥 Hospital Deterioration — Next-12h Early Warning Baseline

Production-minded baseline for predicting **clinical deterioration in the next 12 hours** from a hospital early-warning cohort.

The notebook turns the ML-ready panel into:

- Strong tabular baselines (Logistic / HGB / XGBoost).
- A simple but effective **ensemble**.
- **Threshold & cost** views that are usable in dashboards and alert simulators.
- Exported **model + policy artifacts** ready for integration.

---

## 🎯 Goals

This baseline focuses on:

- Using `hospital_deterioration_ml_ready.csv` as the primary ML table.
- Optionally enriching with **short-window temporal features** from:
  - `vitals_timeseries.csv`
  - `labs_timeseries.csv`
- Training and comparing:
  - **Logistic Regression** (with scaling + one-hot encoding)
  - **HistGradientBoostingClassifier**
  - **XGBoost** (if installed)
- Building a **simple ensemble** over model probabilities.
- Exploring **thresholds and alert policies** under different FN / FP cost setups.
- Saving **models and policy tables** for downstream dashboards / simulations.

---

## 📦 Data & Inputs

Core files expected under `DATA_DIR` (default: current folder):

- `hospital_deterioration_ml_ready.csv`  
  Hour-level ML-ready panel with `deterioration_next_12h` label.

Optional (for temporal enrichment):

- `vitals_timeseries.csv`  
  Hourly vitals per patient (`heart_rate`, `respiratory_rate`, `spo2_pct`, `systolic_bp`, ...).

- `labs_timeseries.csv`  
  Hourly labs (`lactate`, `wbc_count`, `creatinine`, ...).

If `patient_id` and `hour_from_admission` are present in the ML table and  
`USE_TEMPORAL_FEATURES = True`, the notebook:

- Builds **3-hour rolling means** for selected vitals & labs.
- Adds simple **1-hour deltas**.
- Merges them back into the ML-ready table.
- Fills any merge / delta NaNs with `0.0` for robustness.

---

## 🧠 Modeling Approach

**Target**

- `deterioration_next_12h` (binary): 1 = deterioration within the next 12 hours.

**Features**

- All non-ID columns except the target:
  - Numeric features (vitals, labs, risk scores, engineered features, etc.).
  - Categorical features (e.g., device / route / status fields).
- Optional temporal features (rolling means + deltas) if enabled.

**Preprocessing**

- **Linear models (LogReg):**
  - `StandardScaler` for numeric columns.
  - `OneHotEncoder` (dense, `handle_unknown="ignore"`) for categorical columns.

- **Tree models (HGB / XGBoost):**
  - Numeric columns passed **as is**.
  - Same `OneHotEncoder` for categorical columns.

**Models**

- `LogisticRegression`
  - `class_weight="balanced"`
  - `max_iter=1000`

- `HistGradientBoostingClassifier`
  - Lower learning rate
  - Regularised leaf settings for stability

- `XGBClassifier` (optional)
  - Histogram-based tree method
  - Reasonable defaults for n_estimators, max_depth, subsample, colsample

**Evaluation**

- 5-fold **StratifiedKFold** cross-validation on the training set:
  - `ROC AUC`
  - `Average Precision (PR AUC)`

- Final fit on full training set and evaluation on a held-out test set:
  - Test **ROC AUC** and **PR AUC** for each model.
  - **Ensemble** probability = mean of available base model probabilities.

---

## ⚖️ Thresholds & Alert Policies

Using the ensemble scores as the main risk signal:

- Grid of thresholds between 0.10 and 0.90.
- For each threshold:
  - Precision, recall, FPR, alert rate.
  - Counts: TP, FP, FN, TN.
  - Simple cost:  
    `cost = cost_fn * FN + cost_fp * FP`.

Several policy views are computed, e.g.:

- **Balanced policy:** `FN=5`, `FP=1`
- **High-recall policy:** `FN=15`, `FP=1`
- **Low-alerts policy:** `FN=5`, `FP=2`

The notebook:

- Prints best (lowest-cost) rows for each policy.
- Plots **threshold vs recall / precision / alert rate** for the balanced setting.

This gives a clean starting point for picking thresholds under different operational trade-offs.

---

## 📂 Exported Artifacts

All artifacts are saved under:

- `artifacts/`

Content:

- `model_logreg.pkl` — Logistic Regression pipeline.
- `model_hgb.pkl` — HistGradientBoosting pipeline.
- `model_xgb.pkl` (if XGBoost is available).
- `policy_thresholds_balanced.csv` — threshold table for the balanced policy.

These are ready to be:

- Loaded in **Streamlit / dashboard apps**.
- Used in **offline simulators** (e.g., alert burden, what-if analyses).
- Embedded in **batch scoring** or **API services**.

---

## ▶️ How to Run

### Requirements

- Python 3.10+
- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `xgboost` (optional, but recommended)

Install:

```bash
pip install -r requirements.txt
# or
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### Run the notebook

```bash
git clone https://github.com/<tarekmasryo>/hospital-deterioration-next12h-baseline.git
cd hospital-deterioration-next12h-baseline

# Place CSVs under DATA_DIR (default: current folder)

jupyter notebook "Hospital Deterioration — Next-12h Advanced Early Warning Baseline.ipynb"
```

Adjust `DATA_DIR` at the top of the notebook if your data lives elsewhere.
