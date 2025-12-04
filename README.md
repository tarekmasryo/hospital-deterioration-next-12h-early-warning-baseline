# 🏥 Hospital Deterioration — 12h Early Warning ML Baseline

Production-minded baseline for predicting **clinical deterioration in the next 12 hours** from a hospital early-warning cohort.

The notebook turns an ML-ready panel into:

- Strong tabular baselines (Logistic Regression / HistGradientBoosting / XGBoost).
- A simple but effective probability **ensemble**.
- **Threshold and cost** views that can be used directly in dashboards and alert simulators.
- Exported **model and policy artifacts** ready for integration.

---

## 🎯 Goals

This baseline focuses on:

- Using `hospital_deterioration_ml_ready.csv` as the primary ML table.
- Optionally enriching the ML table with **short-window temporal features** from:
  - `vitals_timeseries.csv`
  - `labs_timeseries.csv`
- Training and comparing:
  - **Logistic Regression** (with scaling and one-hot encoding)
  - **HistGradientBoostingClassifier**
  - **XGBClassifier** (if installed)
- Building a **simple ensemble** over model probabilities.
- Exploring **operating thresholds and alert policies** under different FN / FP cost setups.
- Saving **models and policy tables** for downstream dashboards or simulations.

---

## 📦 Data & Inputs

Core files expected under `DATA_DIR` (default: current directory):

- `hospital_deterioration_ml_ready.csv`  
  Hour-level ML-ready panel with `deterioration_next_12h` binary label.

Optional files (for temporal enrichment):

- `vitals_timeseries.csv`  
  Hourly vital signs per patient (`heart_rate`, `respiratory_rate`, `spo2_pct`, `systolic_bp`, ...).

- `labs_timeseries.csv`  
  Hourly lab values (`lactate`, `wbc_count`, `creatinine`, ...).

If `patient_id` and `hour_from_admission` are present in the ML table and  
`USE_TEMPORAL_FEATURES = True`, the notebook:

- Builds **3-hour rolling means** for selected vitals and labs.
- Adds simple **1-hour deltas** per patient.
- Merges these features back into the ML-ready table.
- Fills any merge/delta NaNs with `0.0` for robustness.

---

## 🧠 Modeling Approach

### Target

- `deterioration_next_12h` (binary):  
  - `1` = deterioration within the next 12 hours  
  - `0` = no deterioration in that horizon

### Features

- All non-ID columns except the target:
  - Numeric features (vitals, labs, risk scores, engineered features, etc.).
  - Categorical features (e.g. device, route, or status fields).
- Optional temporal features (rolling means and deltas) if enabled.

### Preprocessing

- **Linear models (Logistic Regression):**
  - `StandardScaler` for numeric columns.
  - `OneHotEncoder` (dense, `handle_unknown="ignore"`) for categorical columns.

- **Tree-based models (HistGradientBoosting / XGBoost):**
  - Numeric columns passed **as is**.
  - Same `OneHotEncoder` for categorical columns.

### Models

- `LogisticRegression`
  - `class_weight="balanced"`
  - `max_iter=1000`

- `HistGradientBoostingClassifier`
  - Lower learning rate
  - Regularised leaf settings for stability on noisy, imbalanced data

- `XGBClassifier` (optional)
  - Histogram-based tree method
  - Reasonable defaults for `n_estimators`, `max_depth`, `subsample`, and `colsample_bytree`

### Evaluation

- 5-fold **StratifiedKFold** cross-validation on the training set:
  - `ROC AUC`
  - `Average Precision (PR AUC)`

- Final fit on the full training set and evaluation on a held-out test set:
  - Test **ROC AUC** and **PR AUC** for each model.
  - **Ensemble** probability = mean of available base-model probabilities.

---

## ⚖️ Thresholds & Alert Policies

Using the ensemble scores as the main risk signal, the notebook evaluates:

- A grid of thresholds between 0.10 and 0.90.
- For each threshold:
  - Precision, recall, false positive rate, alert rate.
  - Counts: TP, FP, FN, TN.
  - Simple cost:  
    `cost = cost_fn * FN + cost_fp * FP`.

Several policy configurations are computed, for example:

- **Balanced policy:** `FN=5`, `FP=1`
- **High-recall policy:** `FN=15`, `FP=1`
- **Low-alerts policy:** `FN=5`, `FP=2`

For each policy, the notebook:

- Builds a policy table over the threshold grid.
- Prints the best (lowest-cost) rows.
- Plots **threshold vs recall / precision / alert rate** for the balanced setting.

This provides a clean starting point for selecting thresholds under different operational trade-offs (e.g. alarm burden vs missed detections).

---

## 📂 Exported Artifacts

All artifacts are saved under:

- `artifacts/`

Contents:

- `model_logreg.pkl`  
  Logistic Regression pipeline (preprocessing + model).

- `model_hgb.pkl`  
  HistGradientBoosting pipeline.

- `model_xgb.pkl`  
  XGBoost pipeline (if XGBoost is available in the environment).

- `policy_thresholds_balanced.csv`  
  Threshold table for the balanced policy.

- `policy_thresholds_high_recall.csv`  
  Threshold table for the high-recall policy.

- `policy_thresholds_low_alerts.csv`  
  Threshold table for the low-alerts policy.

These artifacts can be:

- Loaded in **Streamlit or other dashboards** to visualise risk distributions and alert volumes.
- Used in **offline simulators** (e.g. alert burden, staffing, what-if analyses).
- Embedded in **batch scoring workflows** or **API services**.

---

## ▶️ How to Run

### Requirements

Python environment (3.10+ recommended) with:

- `pandas`, `numpy`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `xgboost` (optional but recommended for the XGB baseline)

Install via:

```bash
pip install -r requirements.txt
```

or, minimally:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost
```

### Running the notebook

1. Place the required CSV files under `DATA_DIR`  
   (default is the current directory, configurable near the top of the notebook).

2. Open the notebook in Jupyter or VS Code:

```bash
jupyter notebook "Hospital Deterioration — 12h Early Warning ML Baseline.ipynb"
```

3. Run the cells from top to bottom.

The notebook will:

- Load the data and (optionally) build temporal features.
- Train and evaluate the baseline models and ensemble.
- Generate threshold–policy tables and plots.
- Export models and policy tables into the `artifacts/` folder.

---

## Disclaimer

This project is intended for **educational and research purposes only**.  
It is **not** a certified medical device and must **not** be used for direct clinical decision-making or bedside care without appropriate validation, governance, and regulatory approval.
