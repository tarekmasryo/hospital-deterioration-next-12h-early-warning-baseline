import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "hospital-deterioration-next-12h-early-warning-baseline.ipynb"


def read_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_required_repository_files_exist():
    required_files = [
        "README.md",
        "CASE_STUDY.md",
        "CHANGELOG.md",
        "LICENSE",
        "requirements.txt",
        "data/raw/README.md",
        "artifacts/README.md",
        "hospital-deterioration-next-12h-early-warning-baseline.ipynb",
    ]

    missing = [name for name in required_files if not (ROOT / name).exists()]
    assert not missing, f"Missing required files: {missing}"


def test_notebook_is_valid_json_and_has_expected_title():
    notebook = read_notebook()

    assert "cells" in notebook
    assert len(notebook["cells"]) >= 20

    first_markdown = next(
        cell for cell in notebook["cells"] if cell.get("cell_type") == "markdown"
    )
    title = "".join(first_markdown.get("source", []))

    assert "Next-12h Hospital Deterioration" in title
    assert "Early-Warning Baseline" in title


def test_notebook_has_no_captured_error_outputs():
    notebook = read_notebook()

    errors = []
    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue

        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                errors.append((index, output.get("ename"), output.get("evalue")))

    assert not errors, f"Notebook contains error outputs: {errors}"


def test_notebook_contains_leakage_aware_patient_split():
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in read_notebook()["cells"]
    )

    expected_terms = [
        "train_patient_ids",
        "valid_patient_ids",
        "test_patient_ids",
        "patient_id",
        "roc_auc_score",
        "average_precision_score",
        "policy_table",
    ]

    missing = [term for term in expected_terms if term not in source]
    assert not missing, f"Notebook is missing expected terms: {missing}"


def test_readme_mentions_required_dataset_files():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    required_dataset_files = [
        "patients.csv",
        "vitals_timeseries.csv",
        "labs_timeseries.csv",
        "hospital_deterioration_hourly_panel.csv",
        "hospital_deterioration_ml_ready.csv",
    ]

    missing = [name for name in required_dataset_files if name not in readme]
    assert not missing, f"README is missing dataset files: {missing}"


def test_raw_data_files_are_not_committed():
    raw_dir = ROOT / "data" / "raw"
    committed_csv_files = sorted(path.name for path in raw_dir.glob("*.csv"))

    assert committed_csv_files == [], f"Raw CSV files should not be committed: {committed_csv_files}"
