# Case Study — Hospital Deterioration Next-12h Early Warning Baseline

## Problem

Hospital deterioration prediction is a high-imbalance alerting task. The challenge is not only to rank risky observations, but also to understand whether the signal is strong enough to support a practical alert policy.

A useful early-warning workflow should make these trade-offs explicit:

- missed deterioration windows,
- false alerts,
- review burden,
- leakage risk,
- threshold selection.

This project uses a synthetic hospital deterioration dataset to build a clean baseline for predicting whether an hourly observation falls within the next 12 hours before deterioration.

## Dataset

The workflow uses five CSV files:

- `patients.csv`
- `vitals_timeseries.csv`
- `labs_timeseries.csv`
- `hospital_deterioration_hourly_panel.csv`
- `hospital_deterioration_ml_ready.csv`

The joined hourly panel is the main modeling table because it preserves `patient_id`, which is required for patient-level splitting.

## Approach

### 1. Validate before modeling

The notebook starts with structural checks:

- patient IDs align across tables,
- hourly keys are unique,
- panel row count matches the total length of stay,
- the next-12h label matches the documented definition,
- oxygen-flow edge cases are inspected in both directions.

This prevents the notebook from looking polished while hiding dataset issues.

### 2. Explore the early-warning signal

The EDA focuses on questions that matter for the modeling task:

- target imbalance,
- deterioration timing,
- baseline risk score behavior,
- vital-sign and lab trajectories,
- pre-event aligned signal movement.

The goal is to validate whether the dataset contains plausible deterioration patterns before fitting a model.

### 3. Use leakage-aware evaluation

The split is performed at the patient level. This avoids training on some hours from a patient and testing on other hours from the same patient.

The model excludes identifiers, future labels, event metadata, and the simulator-generated latent `baseline_risk_score`.

### 4. Train a simple baseline

The baseline uses:

- preprocessing inside an `sklearn` pipeline,
- median imputation for numeric features,
- most-frequent imputation and one-hot encoding for categorical features,
- logistic regression with class balancing.

This is intentionally simple, reproducible, and easy to inspect.

### 5. Convert scores into an alert policy

The notebook reports ranking quality with ROC AUC and Average Precision, then evaluates threshold trade-offs using:

- precision,
- recall,
- F1,
- false positives,
- false negatives,
- alert rate.

Average Precision is emphasized because the positive next-12h rows are imbalanced.

## Key decisions

- Use the hourly panel rather than only the ML-ready table because patient-level splitting requires `patient_id`.
- Keep the first baseline interpretable and reproducible instead of jumping directly to heavier models.
- Treat threshold selection as an operational decision rather than using an arbitrary 0.50 cutoff.
- Present oxygen-flow anomalies as a simulation/documentation edge case, not a modeling blocker.

## Limitations

- The dataset is synthetic and not suitable for clinical deployment.
- The baseline is row-level; patient-level alert burden should be added before any operational prototype.
- Predicted scores should be calibrated before being interpreted as probabilities.
- Additional temporal features and stronger models may improve ranking performance, but they should use the same patient-level split.

## Next steps

- Add rolling-window and volatility features over recent vitals/labs.
- Compare tree-based models under the same split.
- Add calibration curves, Brier score, and calibration-aware threshold discussion.
- Evaluate alerts per patient and per 100 patient-hours.
- Package the final pipeline into a batch scoring function or API-ready component.
