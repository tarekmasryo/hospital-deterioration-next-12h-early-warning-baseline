# 🏥 Hospital Deterioration — Next-12h Early Warning Baseline

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](#)
[![Notebook](https://img.shields.io/badge/Format-Jupyter%20Notebook-orange)](#)
[![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-yellow)](#)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A decision-ready Kaggle and GitHub notebook for a **synthetic hospital deterioration dataset**.  
The project validates dataset structure, explores early-warning signal behavior, trains a leakage-aware baseline model, and turns model scores into practical alert-policy trade-offs.

> ⚠️ This repository is for education, experimentation, and portfolio review only. It is **not** a medical device and must not be used for clinical decision-making.

---

## ✨ What this project provides

- ✅ **Dataset integrity checks** before modeling
- 🧪 **Leakage-aware patient-level splitting**
- 📊 **Focused EDA** for cohort profile, target imbalance, event timing, and signal movement
- 🤖 **Reproducible baseline model** using an `sklearn` pipeline
- 🎚️ **Threshold policy analysis** for alert rate, precision, recall, false positives, and false negatives
- 🧾 **Case study write-up** for quick portfolio/recruiter review

---

## 📌 Repository contents

```text
.
├── .github/workflows/ci.yml
├── artifacts/
│   └── README.md
├── data/
│   └── raw/
│       └── README.md
├── tests/
│   └── test_repo_integrity.py
├── hospital-deterioration-next-12h-early-warning-baseline.ipynb
├── CASE_STUDY.md
├── CHANGELOG.md
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🎯 Project goal

Hospital deterioration modeling is an imbalanced early-warning problem. A useful notebook should not only train a model; it should also answer operational questions:

- How common are next-12h deterioration windows?
- Are the dataset keys, labels, and patient joins structurally valid?
- Which vitals, labs, and static risk factors move before deterioration?
- Can a simple baseline rank high-risk hourly observations well enough to support an alert-policy discussion?
- What happens to precision, recall, and review burden when the alert threshold changes?

---

## 🧠 Methodology

The notebook follows a production-minded workflow:

1. **Path-safe data loading**
   - Kaggle input discovery
   - local `data/raw/` fallback
   - required file validation

2. **Dataset integrity checks**
   - patient ID alignment across tables
   - hourly key uniqueness
   - panel row-count consistency
   - next-12h label recomputation
   - oxygen-flow edge-case inspection

3. **Exploratory analysis**
   - cohort and target profile
   - deterioration timing
   - baseline risk deciles
   - signal trajectories over admission time
   - pre-event aligned trajectories

4. **Leakage-aware baseline modeling**
   - patient-level train/validation/test split
   - preprocessing inside an `sklearn` pipeline
   - logistic regression baseline
   - identifier, event metadata, future labels, and latent simulator score excluded from features

5. **Alert-policy evaluation**
   - validation-selected thresholds
   - test-set precision, recall, F1, alert rate, false positives, and false negatives
   - decision-oriented threshold comparison table

---

## 📊 Dataset files

Place the dataset CSV files under:

```text
data/raw/
```

Required files:

```text
patients.csv
vitals_timeseries.csv
labs_timeseries.csv
hospital_deterioration_hourly_panel.csv
hospital_deterioration_ml_ready.csv
```

Raw data files are intentionally not committed to this repository.

When running on Kaggle, the notebook automatically searches under:

```text
/kaggle/input/
```

No hard-coded Kaggle dataset slug is required.

---

## 🚀 Quick start

### 1) Create a virtual environment

```bash
python -m venv .venv
```

### 2) Activate the environment

```bash
# Windows
.\.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3) Install dependencies

```bash
pip install -r requirements.txt
```

### 4) Add the dataset locally

Copy the required CSV files into:

```text
data/raw/
```

### 5) Run the notebook

Open and run:

```text
hospital-deterioration-next-12h-early-warning-baseline.ipynb
```

---

## ✅ Validation status

The repository has been checked for:

- ✅ valid notebook JSON,
- ✅ executed notebook outputs with no captured error outputs,
- ✅ path-safe data loading,
- ✅ required dataset-file documentation,
- ✅ leakage-aware patient-level split,
- ✅ no `patient_id` leakage into model features,
- ✅ no hard-coded Kaggle dataset slug,
- ✅ English-only notebook and documentation,
- ✅ clean GitHub-ready repository layout.

---

## 📈 Metrics reported

The notebook reports:

- **Average Precision** as the primary ranking metric,
- ROC AUC,
- validation/test positive rates,
- precision-recall curve,
- ROC curve,
- threshold policy table,
- alert rate,
- false positives,
- false negatives,
- coefficient inspection for the logistic baseline.

For this imbalanced alerting setup, **Average Precision** is more informative than accuracy.

---

## 🧩 Suggested extensions

Strong next steps:

- 🔁 add rolling means, deltas, and volatility over the last 3–6 hours,
- 🌲 compare tree-based models under the same patient-level split,
- 🎚️ add calibration analysis before interpreting scores as probabilities,
- 🏥 evaluate patient-level alert burden, not only row-level alert burden,
- ⚙️ package the scoring workflow into a batch job or API-ready component.

---

## 📄 License

MIT License. See [LICENSE](LICENSE) for details.

Copyright © Tarek Masryo.
