from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd

from ml_lab.core.io import ensure_output_dir, write_json
from .api import analyze_agreement
from .selection import ClusteringResult


def _matrix_frame(matrix: np.ndarray, labels: list[str]) -> pd.DataFrame:
    return pd.DataFrame(matrix, index=labels, columns=labels)


def save_results(
    results: Iterable[ClusteringResult],
    output_dir: str | Path,
    *,
    agreement_ignore_noise: bool | None = None,
) -> Path:
    output = ensure_output_dir(output_dir)
    results = list(results)
    if agreement_ignore_noise is None:
        first_stability = next(
            (result.stability_analysis for result in results if result.stability_analysis is not None),
            None,
        )
        agreement_ignore_noise = True if first_stability is None else first_stability.ignore_noise
    agreement = analyze_agreement(results, ignore_noise=agreement_ignore_noise)
    write_json(output / "result.json", [result.to_record() for result in results])
    write_json(output / "algorithm_agreement.json", agreement.summary_record())

    metric_rows = []
    candidate_rows = []
    assignments = {}
    models = output / "models"
    models.mkdir(exist_ok=True)
    stability_dir = output / "stability"
    stability_dir.mkdir(exist_ok=True)
    stability_summary_rows = []
    repeat_agreement_rows = []

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
        if result.stability_analysis is not None:
            row.update({f"repeat_{k}": v for k, v in result.stability_analysis.summary_record().items() if k != "seeds"})
        metric_rows.append(row)
        assignments[result.estimator_id] = result.labels
        joblib.dump(result.fitted_model, models / f"{result.estimator_id}.joblib")

        for candidate in result.candidates:
            candidate_row = {
                "estimator_id": candidate.estimator_id,
                "estimator_name": candidate.estimator_name,
                "params": json.dumps(candidate.params, default=str, sort_keys=True),
                "stability": candidate.stability,
                "stability_std": candidate.stability_std,
                "failed_repeats": candidate.failed_repeats,
                "elapsed_seconds": candidate.elapsed_seconds,
                "warning": candidate.warning,
            }
            candidate_row.update(candidate.metrics)
            candidate_rows.append(candidate_row)

        stability = result.stability_analysis
        if stability is not None:
            summary = {"estimator_id": result.estimator_id, **stability.summary_record()}
            summary["seeds"] = json.dumps(summary["seeds"])
            stability_summary_rows.append(summary)

            repeat_columns = [f"repeat_{index}_seed_{seed}" for index, seed in enumerate(stability.seeds)]
            pd.DataFrame(stability.repeat_labels.T, columns=repeat_columns).to_csv(
                stability_dir / f"{result.estimator_id}_repeat_assignments.csv",
                index_label="row_index",
            )
            pd.DataFrame(stability.consensus_matrix).to_csv(
                stability_dir / f"{result.estimator_id}_consensus.csv",
                index_label="row_index",
            )
            pd.DataFrame(stability.eligible_counts).to_csv(
                stability_dir / f"{result.estimator_id}_consensus_eligible_counts.csv",
                index_label="row_index",
            )
            for record in stability.pairwise_records:
                repeat_agreement_rows.append({"estimator_id": result.estimator_id, **record})

    pd.DataFrame(metric_rows).to_csv(output / "metrics.csv", index=False)
    pd.DataFrame(candidate_rows).to_csv(output / "candidates.csv", index=False)
    pd.DataFrame(assignments).to_csv(output / "cluster_assignments.csv", index_label="row_index")
    pd.DataFrame(stability_summary_rows).to_csv(stability_dir / "summary.csv", index=False)
    pd.DataFrame(repeat_agreement_rows).to_csv(stability_dir / "repeat_agreement.csv", index=False)

    ids = list(agreement.estimator_ids)
    _matrix_frame(agreement.adjusted_rand_matrix, ids).to_csv(
        output / "algorithm_agreement_adjusted_rand.csv",
        index_label="estimator_id",
    )
    _matrix_frame(agreement.normalized_mutual_info_matrix, ids).to_csv(
        output / "algorithm_agreement_nmi.csv",
        index_label="estimator_id",
    )
    _matrix_frame(agreement.coverage_matrix, ids).to_csv(
        output / "algorithm_agreement_coverage.csv",
        index_label="estimator_id",
    )
    pd.DataFrame(agreement.pairwise_records).to_csv(output / "algorithm_agreement_pairs.csv", index=False)
    return output
