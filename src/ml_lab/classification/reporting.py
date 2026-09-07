from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd

from ml_lab.core.io import ensure_output_dir, write_json
from .selection import ClassificationResult


def save_results(results: Iterable[ClassificationResult], output_dir: str | Path) -> Path:
    output = ensure_output_dir(output_dir)
    results = list(results)
    write_json(output / "result.json", [result.to_record() for result in results])

    rows = []
    for result in results:
        row = {
            "task": result.task,
            "estimator_id": result.estimator_id,
            "estimator_name": result.estimator_name,
            "best_cv_score": result.best_cv_score,
            "best_params": json.dumps(result.best_params, default=str, sort_keys=True),
            "search_seconds": result.search_seconds,
            "cv_folds_used": result.cv_folds_used,
            "preprocessing": result.preprocessing,
            "failed_candidates": result.failed_candidates,
        }
        row.update(result.metrics)
        rows.append(row)
    pd.DataFrame(rows).to_csv(output / "metrics.csv", index=False)

    candidate_rows = []
    for result in results:
        for candidate in result.candidates:
            candidate_rows.append({
                "estimator_id": result.estimator_id,
                "estimator_name": result.estimator_name,
                "params": json.dumps(candidate["params"], default=str, sort_keys=True),
                "mean_test_score": candidate["mean_test_score"],
                "std_test_score": candidate["std_test_score"],
                "rank_test_score": candidate["rank_test_score"],
            })
    pd.DataFrame(candidate_rows).to_csv(output / "candidates.csv", index=False)

    models = output / "models"
    models.mkdir(exist_ok=True)
    for result in results:
        joblib.dump(result.fitted_model, models / f"{result.estimator_id}.joblib")
    return output
