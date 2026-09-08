from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from ml_lab.core.io import ensure_output_dir, write_json
from .selection import RegressionResult


def _prediction_frame(result: RegressionResult) -> pd.DataFrame:
    y_true = np.asarray(result.y_true)
    y_pred = np.asarray(result.y_pred)
    if y_true.ndim == 1:
        y_true = y_true.reshape(-1, 1)
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)

    data: dict[str, np.ndarray] = {}
    for index, target in enumerate(result.target_names):
        data[f"{target}__actual"] = y_true[:, index]
        data[f"{target}__predicted"] = y_pred[:, index]
    frame = pd.DataFrame(data)
    frame.insert(0, "estimator_id", result.estimator_id)
    return frame


def save_results(results: Iterable[RegressionResult], output_dir: str | Path) -> Path:
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
            "target_names": json.dumps(result.target_names),
            "multi_output": result.multi_output,
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

    prediction_frames = [_prediction_frame(result) for result in results]
    if prediction_frames:
        pd.concat(prediction_frames, ignore_index=True).to_csv(output / "predictions.csv", index=False)

    models = output / "models"
    models.mkdir(exist_ok=True)
    for result in results:
        joblib.dump(result.fitted_model, models / f"{result.estimator_id}.joblib")
    return output
