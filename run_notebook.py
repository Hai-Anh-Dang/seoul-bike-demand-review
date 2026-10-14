"""Run every cell, or verify saved public evidence with --check without writing."""
import sys
sys.dont_write_bytecode = True

import argparse
import os
import platform
import tempfile
from PIL import Image
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import re
import sys
from urllib.parse import unquote, urlsplit

import nbformat
import numpy as np
import pandas as pd
from packaging.requirements import Requirement
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss,
                             f1_score, precision_score, recall_score, r2_score)

NOTEBOOK = "seoul_bike_machine_learning.ipynb"
PROJECT_KIND = "bike"
EXPECTED_CODE_CELLS = 15
PUBLIC_FIGURES = ['03_model_comparison.png', '04_residual_diagnostics.png']


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _check_relative_links(text, parent, root):
    checked = 0
    for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
        target = target.strip().strip("<>")
        parsed = urlsplit(target)
        if parsed.scheme or target.startswith("#"):
            continue
        path = (parent / unquote(parsed.path)).resolve()
        _require(path.is_relative_to(root), "A Markdown link escapes the project root.")
        _require(path.exists(), f"Missing relative link: {target}")
        checked += 1
    return checked


def validate_project(project_root=None, full_run=False):
    root = Path(project_root or Path(__file__).resolve().parent).resolve()
    notebook_path = root / NOTEBOOK
    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    _require(len(code_cells) == EXPECTED_CODE_CELLS, "Unexpected code-cell count.")
    _require([cell.execution_count for cell in code_cells] == list(range(1, EXPECTED_CODE_CELLS + 1)),
             "Code cells must have complete sequential execution counts.")
    _require(not any(output.output_type == "error" for cell in code_cells for output in cell.outputs), "Notebook contains a cell error.")
    _require(notebook.metadata.project.status == "executed_successfully", "Notebook has no successful execution status.")
    for cell in code_cells:
        compile(cell.source, NOTEBOOK, "exec")

    summary_path = root / "results" / "execution_summary.json"
    if full_run:
        summary = None
        manifest = json.loads((root / ".generated" / "execution_details.json").read_text(encoding="utf-8"))
    else:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _require(summary["status"] == "executed_and_validated" and summary["fresh_execution_claim"],
                 "No completed full-run evidence is recorded.")
        _require(summary["notebook_sha256"] == hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
                 "Notebook changed after the recorded full run; rerun before claiming current outputs.")
        _require(summary["code_cells"] == len(code_cells) and summary["all_code_cells_executed"]
                 and summary["cell_errors"] == 0, "Recorded code-cell checks are incomplete.")
        manifest = summary["analysis"]
    _require(manifest["status"] == "executed_successfully" and manifest["assertions_passed"],
             "Notebook analytical checks did not finish successfully.")

    provenance = json.loads((root / "data" / "provenance.json").read_text(encoding="utf-8"))
    entries = provenance.get("files", [{"name": provenance.get("bundled_file"), "sha256": provenance.get("sha256")}])
    raw_hashes = {}
    for entry in entries:
        path = root / "data" / entry["name"]
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        _require(actual == entry["sha256"], f"Raw-data hash mismatch: {entry['name']}")
        raw_hashes[entry["name"]] = actual

    table = pd.read_csv(root / "results" / "model_metrics.csv")
    validated = []
    prediction_rows = {}
    if PROJECT_KIND == "bike":
        predictions = pd.read_csv(root / "results" / "test_predictions.csv")
        _require(predictions["timestamp"].is_unique, "Duplicate prediction timestamps.")
        timestamps = pd.to_datetime(predictions["timestamp"])
        _require(timestamps.is_monotonic_increasing, "Prediction timestamps are unordered.")
        _require((timestamps >= pd.Timestamp("2018-09-01")).all(), "Test predictions precede the fixed holdout.")
        _require((table["rmse"] + 1e-10 >= table["mae"]).all(), "RMSE is smaller than MAE.")
        raw = pd.read_csv(root / "data" / "SeoulBikeData.csv", encoding="cp1252")
        raw_times = pd.to_datetime(raw["Date"], format="%d/%m/%Y") + pd.to_timedelta(raw["Hour"], unit="h")
        selected = raw["Functioning Day"].eq("Yes") & raw_times.ge("2018-09-01")
        _require(timestamps.equals(raw_times.loc[selected].reset_index(drop=True)), "Final population differs from raw data.")
        actual = predictions["actual_rentals"].to_numpy()
        _require(np.array_equal(actual, raw.loc[selected, "Rented Bike Count"]), "Actual outcomes differ from raw data.")
        expected_models = set(predictions.columns) - {"timestamp", "actual_rentals", "functioning_day"}
        final_rows = table.loc[table["split"].eq("Final test")]
        _require(set(final_rows["model"]) == expected_models and len(final_rows) == 6,
                 "Final metrics do not cover all saved models.")
        for row in table.loc[table["split"].eq("Final test")].itertuples():
            predicted = predictions[row.model].to_numpy()
            _require(np.isfinite(predicted).all() and (predicted >= 0).all(), "Invalid rental predictions.")
            residual = actual - predicted
            expected = {"rmse": float(np.sqrt(np.mean(residual ** 2))),
                        "mae": float(np.mean(np.abs(residual))),
                        "r2": float(r2_score(actual, predicted)),
                        "mean_error_actual_minus_prediction": float(np.mean(residual))}
            for measure, value in expected.items():
                _require(np.isclose(value, getattr(row, measure), rtol=1e-10, atol=1e-10), f"Saved metric mismatch: {row.model}, {measure}")
            _require(row.hours == len(actual), "Metric population differs from predictions.")
            validated.append(row.model)
        prediction_rows["Final test"] = len(actual)
    else:
        blocks = [("Validation", "validation_predictions.csv"), ("Final test", "final_test_predictions.csv")]
        if full_run:
            blocks.insert(0, ("Training", "training_predictions.csv"))
        raw = pd.read_csv(root / "data" / "bank-additional-full.csv", sep=";")
        positions = {"Training": (0, 24712), "Validation": (24712, 32950), "Final test": (32950, 41188)}
        for split, filename in blocks:
            directory = ".generated" if split == "Training" else "results"
            predictions = pd.read_csv(root / directory / filename)
            start, end = positions[split]
            _require(np.array_equal(predictions["record_position"], np.arange(start, end)), "Fixed row split changed.")
            _require(np.array_equal(predictions["actual_subscription"], raw.iloc[start:end]["y"].eq("yes").astype(int)),
                     "Saved actual outcomes differ from raw data.")
            _require(predictions["record_position"].is_unique and predictions["record_position"].is_monotonic_increasing,
                     "Prediction row positions are not unique and ordered.")
            actual = predictions["actual_subscription"].to_numpy()
            subset = table.loc[table["split"].eq(split)]
            expected_models = set(predictions.columns) - {"record_position", "actual_subscription"}
            _require(set(subset["model"]) == expected_models and len(subset) == 3,
                     "Metrics do not cover all saved models.")
            for row in table.loc[table["split"].eq(split)].itertuples():
                probability = predictions[row.model].to_numpy()
                _require(np.isfinite(probability).all() and ((probability >= 0) & (probability <= 1)).all(), "Invalid subscription probabilities.")
                _require(row.threshold == 0.5, "Frozen threshold changed.")
                predicted = probability >= row.threshold
                expected = {"accuracy": accuracy_score(actual, predicted),
                    "precision": precision_score(actual, predicted, zero_division=0),
                    "recall": recall_score(actual, predicted, zero_division=0),
                    "f1": f1_score(actual, predicted, zero_division=0),
                    "average_precision": average_precision_score(actual, probability),
                    "brier_score": brier_score_loss(actual, probability),
                    "positive_prevalence": actual.mean(), "predicted_positive_records": predicted.sum(),
                    "true_negatives": ((actual == 0) & ~predicted).sum(),
                    "false_positives": ((actual == 0) & predicted).sum(),
                    "false_negatives": ((actual == 1) & ~predicted).sum(),
                    "true_positives": ((actual == 1) & predicted).sum()}
                for measure, value in expected.items():
                    _require(np.isclose(value, getattr(row, measure), rtol=1e-10, atol=1e-10), f"Saved metric mismatch: {split}, {row.model}, {measure}")
                _require(row.records == len(actual), "Metric population differs from predictions.")
                validated.append(f"{split}: {row.model}")
            prediction_rows[split] = len(actual)

    dependency_count = 0
    dependencies = {}
    for filename in ["requirements.txt"]:
        for line in (root / filename).read_text(encoding="utf-8").splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            requirement = Requirement(line)
            if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
                _require(version(requirement.name) in requirement.specifier, f"Installed version differs from pin: {requirement.name}")
                dependency_count += 1
                dependencies[requirement.name] = version(requirement.name)

    relative_links = 0
    for path in root.rglob("*.md"):
        if not any(part in [".git", ".venv", "__pycache__", ".generated"] for part in path.relative_to(root).parts):
            relative_links += _check_relative_links(path.read_text(encoding="utf-8"), path.parent, root)
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            relative_links += _check_relative_links(cell.source, root, root)

    figures = ["results/figures/" + name for name in PUBLIC_FIGURES]
    for figure in figures:
        with Image.open(root / figure) as image:
            _require(image.format == "PNG", "Retained chart is not a PNG.")
            image.verify()
    embedded_charts = sum("image/png" in output.get("data", {}) for cell in code_cells for output in cell.outputs)
    _require(embedded_charts == 6, "Expected six embedded charts.")
    if full_run:
        all_figures = [cell.metadata.get("figure_file", cell.metadata.get("generated_figure"))
                       for cell in notebook.cells if cell.metadata.get("figure_file") or cell.metadata.get("generated_figure")]
        _require(len(all_figures) == 6 and all((root / name).is_file() for name in all_figures),
                 "A full run must regenerate all six chart files.")
    public_artifacts = ["results/model_metrics.csv", *figures]
    public_artifacts += (["results/test_predictions.csv"] if PROJECT_KIND == "bike" else
                         ["results/validation_predictions.csv", "results/final_test_predictions.csv"])
    artifact_hashes = {name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in public_artifacts}
    if summary is not None:
        _require(summary["raw_hashes_verified"] == raw_hashes, "Recorded raw hashes changed.")
        _require(summary["dependencies_verified"] == dependencies, "Recorded dependency versions changed.")
        _require(summary["public_artifact_sha256"] == artifact_hashes, "Saved public artifact bytes changed.")
        _require(summary["metrics"] == json.loads(table.to_json(orient="records", double_precision=15)),
                 "Saved metrics differ from the execution summary.")
        expected_rows = {"Final test": 1937} if PROJECT_KIND == "bike" else {"Training": 24712, "Validation": 8238, "Final test": 8238}
        _require(summary["full_run_checks"]["prediction_rows"] == expected_rows,
                 "Recorded full-run prediction verification is incomplete.")
        _require(summary["full_run_checks"]["all_six_chart_files_verified"], "Recorded full-run chart verification is incomplete.")
    for path in root.rglob("*"):
        if not path.is_file() or any(part in [".git", ".venv", "__pycache__", ".generated"] for part in path.relative_to(root).parts):
            continue
        _require(not path.name.startswith(("business_structure", "comment_only", "_stage_projects")), "Historical or staging file is included.")
        _require(path.suffix.lower() not in [".docx", ".qmd", ".zip", ".bak"], "An excluded source/archive file is included.")
        if path.suffix in [".md", ".py", ".json"]:
            text = path.read_text(encoding="utf-8")
            _require(re.search(r"(?<![\w])[A-Za-z]:[\\/]", text) is None, "An absolute Windows path is included.")
            if path.suffix in [".md", ".json"]:
                _require("parent portfolio" not in text.lower(), "Parent-project instructions remain.")
    source_text = "\n".join(cell.source for cell in code_cells)
    _require("01_seoul_bike_machine_learning" not in source_text and "03_bank_marketing_leakage" not in source_text,
             "A numbered parent-folder fallback remains.")
    _require(b"\r\n" not in notebook_path.read_bytes(), "Notebook must use LF line endings for stable Git checkout.")
    return {"status": "passed", "validation_mode": "full_run" if full_run else "read_only_public_artifacts",
        "validated_at_utc": datetime.now(timezone.utc).isoformat(), "notebook": NOTEBOOK,
        "notebook_sha256": hashlib.sha256(notebook_path.read_bytes()).hexdigest(),
        "code_cells": len(code_cells), "all_code_cells_executed": True, "cell_errors": 0, "nbformat_valid": True,
        "dependencies_verified": dependencies, "public_artifact_sha256": artifact_hashes,
        "embedded_charts_verified": embedded_charts, "all_six_chart_files_verified": full_run,
        "prediction_scope": "All three blocks" if full_run and PROJECT_KIND == "bank" else
            "Final test only" if PROJECT_KIND == "bike" else "Validation and Final test only",
        "scope_note": (
            "Saved public checks do not refit or require .generated. Both modes verify the same final regression predictions; a full run also regenerates and checks the optional diagnostics."
            if PROJECT_KIND == "bike" else
            "Saved public checks do not refit or require .generated. A full run additionally verifies training probabilities and regenerates the optional diagnostics."),
        "model_execution_status": manifest["status"], "raw_hashes_verified": raw_hashes,
        "saved_metric_comparisons_verified": validated, "prediction_rows": prediction_rows,
        "relative_links_verified": relative_links, "active_locked_dependencies_verified": dependency_count,
        "chart_files_verified": len(figures), "standalone_paths": True,
        "scope": "Saved execution and internal consistency; not deployment or business-value approval"}



def execute_notebook():
    from nbclient import NotebookClient
    from jupyter_client import KernelManager
    from jupyter_client.kernelspec import KernelSpecManager

    root = Path(__file__).resolve().parent
    path = root / NOTEBOOK
    summary_path = root / "results" / "execution_summary.json"
    details_path = root / ".generated" / "execution_details.json"
    summary_path.parent.mkdir(exist_ok=True)
    details_path.parent.mkdir(exist_ok=True)
    # A failed rerun must not preserve an earlier completion record.
    summary_path.unlink(missing_ok=True)
    details_path.unlink(missing_ok=True)
    notebook = nbformat.read(path, as_version=4)
    notebook.metadata.project.status = "execution_in_progress"
    completed = False
    try:
        with tempfile.TemporaryDirectory(prefix=f"{PROJECT_KIND}_portfolio_kernel_") as temporary:
            kernel_root = Path(temporary)
            spec_dir = kernel_root / "portfolio_python"
            spec_dir.mkdir()
            (spec_dir / "kernel.json").write_text(json.dumps({
                "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}",
                         "--HistoryManager.hist_file=:memory:"],
                "display_name": "Portfolio Python", "language": "python"}), encoding="utf-8")
            manager = KernelManager(kernel_name="portfolio_python",
                kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_root)]))
            client = NotebookClient(notebook, km=manager, timeout=1200,
                resources={"metadata": {"path": str(root)}}, allow_errors=False)
            try:
                client.execute(env={**os.environ, "IPYTHONDIR": str(kernel_root / "ipython"),
                    "MPLCONFIGDIR": str(kernel_root / "matplotlib"), "PYTHONDONTWRITEBYTECODE": "1"})
            finally:
                if manager.has_kernel:
                    manager.shutdown_kernel(now=True)
        notebook.metadata.project.status = "executed_successfully"
        nbformat.validate(notebook)
        with path.open("w", encoding="utf-8", newline="\n") as output:
            nbformat.write(notebook, output)
        report = validate_project(root, full_run=True)
        details = json.loads(details_path.read_text(encoding="utf-8"))
        _require(details["python_version"] == platform.python_version(), "Kernel interpreter differs from runner.")
        for name, installed in details["dependencies"].items():
            _require(version(name) == installed, f"Kernel dependency differs: {name}")
        table = pd.read_csv(root / "results" / "model_metrics.csv")
        summary = {**report, "status": "executed_and_validated", "fresh_execution_claim": True,
            "python_version": platform.python_version(), "kernel_uses_current_interpreter": True,
            "analysis": details, "metrics": json.loads(table.to_json(orient="records", double_precision=15)),
            "full_run_checks": report,
            "saved_check_scope": "Final test only" if PROJECT_KIND == "bike" else "Validation and Final test only"}
        # Record the final notebook hash without changing notebook bytes afterward.
        summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
        validate_project(root)
        completed = True
        return summary
    finally:
        if not completed:
            summary_path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true",
                        help="Read-only verification of saved public artifacts; no model refit.")
    arguments = parser.parse_args()
    report = validate_project() if arguments.check else execute_notebook()
    if arguments.check:
        report["fresh_execution_claim"] = False
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Validation failed: {error}", file=sys.stderr)
        raise SystemExit(1)
