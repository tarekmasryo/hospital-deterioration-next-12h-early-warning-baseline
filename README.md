# 🏥 Hospital Deterioration — 12h Early Warning Baseline

A production-minded baseline notebook for predicting **clinical deterioration in the next 12 hours** from an ML-ready hospital cohort.

Notebook: `hospital-deterioration-next-12h-early-warning-baseline.ipynb`  
Case study: `CASE_STUDY.md`

---

## What this repository includes
- A single notebook with EDA, modeling, calibration, and threshold policies
- Exported artifacts under `./artifacts/` (models + threshold policy tables)

---

## Dataset
Place your CSV files under `data/raw/` (recommended):

Required
- `hospital_deterioration_ml_ready.csv`

Optional (enables temporal-window features)
- `vitals_timeseries.csv`
- `labs_timeseries.csv`

See `data/raw/README.md`.

The notebook also attempts to locate files under Kaggle inputs when running on Kaggle.

---

## Quick start

### 1) Install
```bash
python -m venv .venv
# Windows: .\.venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Run the notebook
Open and run:
- `hospital-deterioration-next-12h-early-warning-baseline.ipynb`

Outputs are exported to:
- `artifacts/`

See `artifacts/README.md`.

---

## Outputs you get
The notebook produces decision-ready deliverables:
- baseline models (CPU-friendly)
- calibrated risk scores (when supported by the model)
- threshold policy tables for different alerting regimes:
  - balanced
  - high recall
  - low alerts

---

## Disclaimer
This project is intended for educational and research use. It is **not** a certified medical device and must not be used for clinical decision-making without appropriate validation, governance, and regulatory review.
