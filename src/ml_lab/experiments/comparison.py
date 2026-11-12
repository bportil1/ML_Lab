"""S5.1 read-only, artifact-verified comparison of experiment runs.

This module does not rerun a model, normalize metrics across model families,
select winners, or overwrite any stored result. Only pinned experiment snapshots
and validated, project-local result artifacts are read.
"""
from __future__ import annotations

import csv
import hashlib
from io import StringIO
import json
import math
from typing import Any, Mapping, Sequence

from .optimization import OPTIMIZER_EXECUTOR
from .store import ExperimentStore, ExperimentStoreError

COMPARISON_SCHEMA = "ml-lab.run-comparison@1"
TRAINING_EXECUTOR = "ml_lab.experiments.training@1"
_EXECUTORS = frozenset({TRAINING_EXECUTOR, OPTIMIZER_EXECUTOR})
_MAX_RUNS = 25
_MAX_RESULT_BYTES = 5 * 1024 * 1024


def _number(value: Any) -> int | float | None:
    """Retain only finite, native JSON numeric metrics, excluding booleans."""
    if type(value) not in (int, float) or not math.isfinite(value):
        return None
    return value


def _model_metrics(document: Any) -> dict[str, int | float]:
    if not isinstance(document, Mapping):
        return {}
    # Single-result representation tasks use ``result``; suites use
    # ``results`` with distinct estimator IDs. Avoid arbitrary nested scans.
    candidates = document.get("results")
    if not isinstance(candidates, list):
        candidates = [document.get("result")]
    extracted: dict[str, int | float] = {}
    for index, result in enumerate(candidates[:32]):
        if not isinstance(result, Mapping):
            continue
        metric_map = result.get("metrics")
        if not isinstance(metric_map, Mapping):
            continue
        estimator = result.get("estimator_id") or result.get("method") or "model"
        if not isinstance(estimator, str) or not estimator or len(estimator) > 80:
            estimator = f"model_{index+1}"
        for key, raw in metric_map.items():
            if not isinstance(key, str) or not key or len(key) > 80:
                continue
            number = _number(raw)
            if number is not None:
                extracted[f"{estimator} · {key}"] = number
    return extracted


def _optimization_metrics(document: Any, plan: Mapping[str, Any]) -> tuple[dict[str, int | float], dict[str, Any]]:
    if not isinstance(document, Mapping) or document.get("algorithm") not in {"firefly", "random_search"}:
        return {}, {}
    metric = plan.get("metric")
    goal = plan.get("goal")
    if not isinstance(metric, str) or goal not in ("minimize", "maximize"):
        return {}, {}
    # Use the optimizer's *observed* objective, not its goal-adjusted fitness.
    # Older records may lack objective terms; never present fitness as an
    # observed model metric without an explicit conversion contract.
    terms = document.get("best_objective_terms")
    observed = _number(terms.get(metric)) if isinstance(terms, Mapping) else None
    values = {f"best observed · {metric}": observed} if observed is not None else {}
    parameters = document.get("best_parameters")
    return values, {
        "algorithm": document.get("algorithm"), "metric": metric, "goal": goal,
        "evaluations": document.get("evaluations") if type(document.get("evaluations")) is int else None,
        "best_parameters": parameters if isinstance(parameters, dict) else {},
    }


def _result_document(store: ExperimentStore, run_id: str) -> tuple[Any | None, str]:
    artifacts = [a for a in store.artifacts.list_for_run(run_id, limit=500)
                 if a["role"] == "output" and a["kind"] == "result"]
    if len(artifacts) != 1:
        return None, "Missing or ambiguous result artifact"
    artifact = artifacts[0]
    if artifact["size_bytes"] > _MAX_RESULT_BYTES:
        return None, "Result exceeds comparison size limit"
    try:
        if store.artifacts.verify(artifact["artifact_id"])["status"] != "valid":
            return None, "Result integrity check failed"
        # Project-relative and symlink-resistant path resolution is provided
        # by the same registry that validated this artifact at registration.
        path, _ = store.artifacts._project_file(store.project_root / artifact["path"])
        if path.stat().st_size > _MAX_RESULT_BYTES:
            return None, "Result exceeds comparison size limit"
        with path.open("rb") as stream:
            raw = stream.read(_MAX_RESULT_BYTES + 1)
        if len(raw) > _MAX_RESULT_BYTES:
            return None, "Result exceeds comparison size limit"
        # Verify the exact bytes parsed, not only a prior on-disk fingerprint.
        if hashlib.sha256(raw).hexdigest() != artifact["sha256"]:
            return None, "Result integrity check failed"
        return json.loads(raw), "ok"
    except (OSError, UnicodeDecodeError, ValueError, TypeError, ExperimentStoreError):
        return None, "Result is unreadable or invalid"


def compare_experiment_runs(
    store: ExperimentStore, experiment_id: str, *,
    run_ids: Sequence[str] | None = None, limit: int = _MAX_RUNS,
) -> dict[str, Any]:
    """Return a non-mutating comparison bound to one experiment and project.

    An explicit set of IDs must all belong to this experiment; no mixed-project
    or arbitrary historic task records are accepted. Selection order is kept.
    """
    experiment = store.get_experiment(experiment_id)
    if type(limit) is not int or not 1 <= limit <= _MAX_RUNS:
        raise ExperimentStoreError("Compare at most 25 runs")
    if run_ids is None:
        runs = [r for r in store.list_runs(experiment_id, limit=100)
                if r["environment"].get("executor") in _EXECUTORS][:limit]
    else:
        if isinstance(run_ids, (str, bytes)) or not isinstance(run_ids, Sequence) or not 1 <= len(run_ids) <= _MAX_RUNS:
            raise ExperimentStoreError("Choose between 1 and 25 run identifiers")
        if any(not isinstance(rid, str) for rid in run_ids) or len(set(run_ids)) != len(run_ids):
            raise ExperimentStoreError("Run identifiers must be unique strings")
        runs = [store.get_run(rid) for rid in run_ids]
        if any(r["experiment_id"] != experiment_id or r["environment"].get("executor") not in _EXECUTORS for r in runs):
            raise ExperimentStoreError("Cannot compare unrelated or unsupported runs")
    rows: list[dict[str, Any]] = []
    keys: set[str] = set()
    for run in runs:
        snapshot = store.get_experiment(experiment_id, revision=run["revision"])
        spec = snapshot["model_spec"]
        dataset = snapshot["dataset_ref"]
        executor = run["environment"].get("executor")
        kind = "optimization" if executor == OPTIMIZER_EXECUTOR else "model"
        cohort = (kind, spec.get("family"), spec.get("method"), dataset.get("sha256"),
                  tuple(dataset.get("feature_columns", ())), tuple(dataset.get("target_columns", ())))
        status_note = "Not complete" if run["status"] != "succeeded" else ""
        metrics: dict[str, int | float] = {}
        optimization: dict[str, Any] = {}
        if run["status"] == "succeeded":
            document, status_note = _result_document(store, run["run_id"])
            if status_note == "ok":
                if kind == "optimization":
                    metrics, optimization = _optimization_metrics(document, run["environment"].get("plan", {}))
                else:
                    metrics = _model_metrics(document)
                if not metrics:
                    status_note = "No supported scalar evaluation metrics"
        keys.update(metrics)
        rows.append({
            "run_id": run["run_id"], "revision": run["revision"],
            "status": run["status"], "created_at": run["created_at"],
            "type": kind, "family": spec.get("family", ""), "method": spec.get("method", ""),
            "dataset_sha256": dataset.get("sha256", ""),
            "feature_columns": dataset.get("feature_columns", []),
            "target_columns": dataset.get("target_columns", []),
            "metrics": metrics, "optimization": optimization,
            "result_note": status_note, "error": run.get("error"),
            "_cohort": cohort,
        })
    cohorts = {row.pop("_cohort") for row in rows}
    return {
        "schema": COMPARISON_SCHEMA, "experiment_id": experiment_id,
        "experiment_name": experiment["name"], "runs": rows,
        "metric_columns": sorted(keys),
        "comparable": len(cohorts) <= 1,
        "warning": ("Runs differ in type, model family/method, dataset content, or selected columns. "
                    "Metrics must not be ranked across these cohorts."
                    if len(cohorts) > 1 else ""),
    }


def _csv_text(value: Any) -> Any:
    """Avoid spreadsheet formula execution when a CSV is opened interactively."""
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@")):
        return "'" + value
    return value


def comparison_csv(report: Mapping[str, Any]) -> str:
    """Export measured values as CSV; preserve unavailable cells as blanks."""
    stream = StringIO(newline="")
    writer = csv.writer(stream)
    metrics = report["metric_columns"]
    writer.writerow(["run_id", "revision", "type", "status", "family", "method",
                     "dataset_sha256", "created_at", "result_note", "error", *metrics])
    for row in report["runs"]:
        writer.writerow([_csv_text(x) for x in (
            row["run_id"], row["revision"], row["type"], row["status"],
            row["family"], row["method"], row["dataset_sha256"],
            row["created_at"], row["result_note"], row["error"] or "",
            *(row["metrics"].get(key, "") for key in metrics),
        )])
    return stream.getvalue()
