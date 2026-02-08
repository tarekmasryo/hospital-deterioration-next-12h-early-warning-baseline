# Case Study — Hospital Deterioration Next 12 Hours (Early Warning Baseline)

## Overview
This project provides a decision-ready baseline for predicting **clinical deterioration in the next 12 hours** from a hospital cohort.

The key deliverables are:
- calibrated risk scores suitable for alert thresholds
- threshold policies that make trade-offs explicit (missed events vs alert volume)
- exportable artifacts (models + policy tables) for downstream dashboards or simulations

## The real problem
Early warning systems have asymmetric costs:
- **False negatives** can delay escalation and increase clinical risk.
- **False positives** create alert fatigue and burn staff attention.

A model metric alone is not an operating plan. The workflow must produce a **clear threshold policy** that matches capacity and risk tolerance.

## Goals (definition of done)
**Functional goals**
- train strong tabular baselines on an ML-ready panel
- output calibrated probabilities where possible
- provide threshold views for different regimes (balanced, high recall, low alerts)

**Engineering goals**
- leak-safe evaluation (no fitting on the holdout)
- reproducible training (seeded splits)
- exported artifacts for downstream use

## Approach
### 1) Baselines that fit clinical ops constraints
The notebook trains multiple CPU-friendly baselines (e.g., Logistic Regression, HistGradientBoosting).
If XGBoost is available in the environment, it can be included as an optional stronger baseline.

### 2) Probability calibration and decision thresholds
The workflow emphasizes probabilities and thresholds:
- calibration improves the meaning of risk scores
- thresholds convert scores into actions under a specific operating regime

### 3) Outputs for decision-making
The notebook exports:
- trained model artifacts
- policy tables with thresholds for different alerting regimes

These outputs are designed to plug into:
- alert simulators
- ward/unit dashboards
- retrospective evaluations

## Limitations
- This is a baseline workflow template, not a certified medical device.
- Real deployment requires governance: clinical validation, monitoring, and regulatory review where applicable.
- Performance and calibration can drift across sites, seasons, and patient mixes.

## Next steps
- add slice reporting (ward, age band, comorbidity groups) for fairness and safety checks
- evaluate alert burden explicitly (alerts per 100 patient-hours)
- monitor calibration drift and retrain triggers over time
