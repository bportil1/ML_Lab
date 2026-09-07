from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

from ml_lab.core.io import ensure_output_dir, write_json
from .selection import ClusteringResult


def save_results(results: Iterable[ClusteringResult], output_dir: str | Path) -> Path:
    output = ensure_output_dir(output_dir)
    results = list(results)
    write_json(output / "result.json", [result.to_record() for result in results])

    metric_rows = []
    candidate_rows = []
    assignments = {}
    models = output / "models"
    models.mkdir(exist_ok=True)

    for result in results:
        row = {
            "task": result.task,
            "estimator_id": result.estimator_id,
            "estimator_name": result.estimator_name,
            "best_params": json.dumps(result.best_params, default=str, sort_keys=True),
            "selection_metric": result.selection_metric,
            "selection_score": result.selection_score,
            "stability": result.stability,
            "preprocessing": result.preprocessing,
        }
        row.update(result.metrics)
        if result.external_metrics:
            row.update({f"external_{k}": v for k, v in result.external_metrics.items()})
        metric_rows.append(row)
        assignments[result.estimator_id] = result.labels
        joblib.dump(result.fitted_model, models / f"{result.estimator_id}.joblib")

        for candidate in result.candidates:
            candidate_row = {
                "estimator_id": candidate.estimator_id,
                "estimator_name": candidate.estimator_name,
                "params": json.dumps(candidate.params, default=str, sort_keys=True),
                "stability": candidate.stability,
                "failed_repeats": candidate.failed_repeats,
                "elapsed_seconds": candidate.elapsed_seconds,
                "warning": candidate.warning,
            }
            candidate_row.update(candidate.metrics)
            candidate_rows.append(candidate_row)

    pd.DataFrame(metric_rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(output / "candidates.csv", index=False)
    pd.DataFrame(assignments).to_csv(output / "cluster_assignments.csv", index_label="row_index")
    return output
