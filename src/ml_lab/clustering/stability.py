from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Iterable, Sequence

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


@dataclass(slots=True)
class RepeatStabilityReport:
    repeats_requested: int
    repeats_succeeded: int
    failed_repeats: int
    seeds: tuple[int, ...]
    ignore_noise: bool
    adjusted_rand_mean: float | None
    adjusted_rand_std: float | None
    adjusted_rand_min: float | None
    adjusted_rand_max: float | None
    normalized_mutual_info_mean: float | None
    normalized_mutual_info_std: float | None
    normalized_mutual_info_min: float | None
    normalized_mutual_info_max: float | None
    consensus_consistency: float | None
    consensus_coverage: float
    repeat_labels: np.ndarray
    consensus_matrix: np.ndarray
    eligible_counts: np.ndarray
    pairwise_records: list[dict[str, Any]]

    def summary_record(self) -> dict[str, Any]:
        return {
            "repeats_requested": self.repeats_requested,
            "repeats_succeeded": self.repeats_succeeded,
            "failed_repeats": self.failed_repeats,
            "seeds": list(self.seeds),
            "ignore_noise": self.ignore_noise,
            "adjusted_rand_mean": self.adjusted_rand_mean,
            "adjusted_rand_std": self.adjusted_rand_std,
            "adjusted_rand_min": self.adjusted_rand_min,
            "adjusted_rand_max": self.adjusted_rand_max,
            "normalized_mutual_info_mean": self.normalized_mutual_info_mean,
            "normalized_mutual_info_std": self.normalized_mutual_info_std,
            "normalized_mutual_info_min": self.normalized_mutual_info_min,
            "normalized_mutual_info_max": self.normalized_mutual_info_max,
            "consensus_consistency": self.consensus_consistency,
            "consensus_coverage": self.consensus_coverage,
        }


@dataclass(slots=True)
class AlgorithmAgreementReport:
    estimator_ids: tuple[str, ...]
    ignore_noise: bool
    adjusted_rand_matrix: np.ndarray
    normalized_mutual_info_matrix: np.ndarray
    coverage_matrix: np.ndarray
    pairwise_records: list[dict[str, Any]]

    def summary_record(self) -> dict[str, Any]:
        valid_pairs = [record for record in self.pairwise_records if record["adjusted_rand"] is not None]
        if valid_pairs:
            mean_ari = float(np.mean([record["adjusted_rand"] for record in valid_pairs]))
            mean_nmi = float(np.mean([record["normalized_mutual_info"] for record in valid_pairs]))
            mean_coverage = float(np.mean([record["coverage_fraction"] for record in valid_pairs]))
        else:
            mean_ari = None
            mean_nmi = None
            mean_coverage = 0.0
        return {
            "estimator_ids": list(self.estimator_ids),
            "ignore_noise": self.ignore_noise,
            "pair_count": len(self.pairwise_records),
            "valid_pair_count": len(valid_pairs),
            "mean_adjusted_rand": mean_ari,
            "mean_normalized_mutual_info": mean_nmi,
            "mean_coverage_fraction": mean_coverage,
        }


def _metric_summary(values: Sequence[float]) -> tuple[float | None, float | None, float | None, float | None]:
    if not values:
        return None, None, None, None
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if not finite.size:
        return None, None, None, None
    return (
        float(np.mean(finite)),
        float(np.std(finite)),
        float(np.min(finite)),
        float(np.max(finite)),
    )


def _agreement_view(a: np.ndarray, b: np.ndarray, *, ignore_noise: bool) -> tuple[np.ndarray, np.ndarray, float]:
    if a.shape != b.shape:
        raise ValueError("cluster label arrays must have identical shapes")
    if not ignore_noise:
        return a, b, 1.0 if a.size else 0.0
    keep = (a != -1) & (b != -1)
    coverage = float(np.mean(keep)) if keep.size else 0.0
    return a[keep], b[keep], coverage


def pairwise_label_agreement(
    a: Any,
    b: Any,
    *,
    ignore_noise: bool = True,
) -> dict[str, float | int | None]:
    left = np.asarray(a)
    right = np.asarray(b)
    left_eval, right_eval, coverage = _agreement_view(left, right, ignore_noise=ignore_noise)
    if left_eval.size < 2:
        return {
            "adjusted_rand": None,
            "normalized_mutual_info": None,
            "coverage_fraction": coverage,
            "sample_count": int(left_eval.size),
        }
    return {
        "adjusted_rand": float(adjusted_rand_score(left_eval, right_eval)),
        "normalized_mutual_info": float(normalized_mutual_info_score(left_eval, right_eval)),
        "coverage_fraction": coverage,
        "sample_count": int(left_eval.size),
    }


def coassignment_matrix(
    label_runs: Iterable[Any],
    *,
    ignore_noise: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    runs = [np.asarray(labels) for labels in label_runs]
    if not runs:
        return np.empty((0, 0), dtype=float), np.empty((0, 0), dtype=int)
    size = len(runs[0])
    if any(len(labels) != size for labels in runs):
        raise ValueError("all repeated clustering label arrays must have the same length")

    same_counts = np.zeros((size, size), dtype=float)
    eligible_counts = np.zeros((size, size), dtype=int)
    for labels in runs:
        if ignore_noise:
            valid = labels != -1
        else:
            valid = np.ones(size, dtype=bool)
        eligible = valid[:, None] & valid[None, :]
        same = (labels[:, None] == labels[None, :]) & eligible
        same_counts += same.astype(float)
        eligible_counts += eligible.astype(int)

    consensus = np.full((size, size), np.nan, dtype=float)
    np.divide(same_counts, eligible_counts, out=consensus, where=eligible_counts > 0)
    return consensus, eligible_counts


def consensus_consistency(consensus: Any) -> float | None:
    matrix = np.asarray(consensus, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("consensus matrix must be square")
    if matrix.shape[0] < 2:
        return None
    upper = matrix[np.triu_indices(matrix.shape[0], k=1)]
    finite = upper[np.isfinite(upper)]
    if not finite.size:
        return None
    # A pair is stable whether it is consistently together (p≈1) or apart (p≈0).
    return float(np.mean(np.maximum(finite, 1.0 - finite)))


def repeat_stability_report(
    label_runs: Iterable[Any],
    *,
    repeats_requested: int | None = None,
    seeds: Iterable[int] | None = None,
    failed_repeats: int = 0,
    ignore_noise: bool = True,
) -> RepeatStabilityReport:
    runs = [np.asarray(labels) for labels in label_runs]
    if runs and any(labels.shape != runs[0].shape for labels in runs):
        raise ValueError("all repeated clustering label arrays must have identical shapes")
    repeat_array = np.stack(runs, axis=0) if runs else np.empty((0, 0), dtype=int)
    seed_values = tuple(int(seed) for seed in (seeds or range(len(runs))))
    if len(seed_values) != len(runs):
        raise ValueError("seeds must contain one value per successful label run")

    pairwise_records: list[dict[str, Any]] = []
    ari_values: list[float] = []
    nmi_values: list[float] = []
    for (index_a, labels_a), (index_b, labels_b) in combinations(enumerate(runs), 2):
        metrics = pairwise_label_agreement(labels_a, labels_b, ignore_noise=ignore_noise)
        record = {
            "repeat_a": index_a,
            "repeat_b": index_b,
            "seed_a": seed_values[index_a],
            "seed_b": seed_values[index_b],
            **metrics,
        }
        pairwise_records.append(record)
        if metrics["adjusted_rand"] is not None:
            ari_values.append(float(metrics["adjusted_rand"]))
        if metrics["normalized_mutual_info"] is not None:
            nmi_values.append(float(metrics["normalized_mutual_info"]))

    consensus, eligible_counts = coassignment_matrix(runs, ignore_noise=ignore_noise)
    if eligible_counts.size:
        possible = len(runs) * eligible_counts.shape[0] * eligible_counts.shape[1]
        coverage = float(eligible_counts.sum() / possible) if possible else 0.0
    else:
        coverage = 0.0
    ari_mean, ari_std, ari_min, ari_max = _metric_summary(ari_values)
    nmi_mean, nmi_std, nmi_min, nmi_max = _metric_summary(nmi_values)
    requested = int(repeats_requested if repeats_requested is not None else len(runs) + failed_repeats)
    return RepeatStabilityReport(
        repeats_requested=requested,
        repeats_succeeded=len(runs),
        failed_repeats=int(failed_repeats),
        seeds=seed_values,
        ignore_noise=bool(ignore_noise),
        adjusted_rand_mean=ari_mean,
        adjusted_rand_std=ari_std,
        adjusted_rand_min=ari_min,
        adjusted_rand_max=ari_max,
        normalized_mutual_info_mean=nmi_mean,
        normalized_mutual_info_std=nmi_std,
        normalized_mutual_info_min=nmi_min,
        normalized_mutual_info_max=nmi_max,
        consensus_consistency=consensus_consistency(consensus),
        consensus_coverage=coverage,
        repeat_labels=repeat_array,
        consensus_matrix=consensus,
        eligible_counts=eligible_counts,
        pairwise_records=pairwise_records,
    )


def algorithm_agreement_report(
    estimator_ids: Sequence[str],
    labels: Sequence[Any],
    *,
    ignore_noise: bool = True,
) -> AlgorithmAgreementReport:
    ids = tuple(str(estimator_id) for estimator_id in estimator_ids)
    label_arrays = [np.asarray(values) for values in labels]
    if len(ids) != len(label_arrays):
        raise ValueError("estimator_ids and labels must have the same length")
    if label_arrays and any(array.shape != label_arrays[0].shape for array in label_arrays):
        raise ValueError("all algorithm label arrays must have identical shapes")

    count = len(ids)
    ari = np.full((count, count), np.nan, dtype=float)
    nmi = np.full((count, count), np.nan, dtype=float)
    coverage = np.zeros((count, count), dtype=float)
    records: list[dict[str, Any]] = []

    for index in range(count):
        valid = label_arrays[index] != -1 if ignore_noise else np.ones_like(label_arrays[index], dtype=bool)
        diag_coverage = float(np.mean(valid)) if valid.size else 0.0
        ari[index, index] = 1.0 if valid.any() else np.nan
        nmi[index, index] = 1.0 if valid.any() else np.nan
        coverage[index, index] = diag_coverage

    for left, right in combinations(range(count), 2):
        metrics = pairwise_label_agreement(label_arrays[left], label_arrays[right], ignore_noise=ignore_noise)
        ari_value = metrics["adjusted_rand"]
        nmi_value = metrics["normalized_mutual_info"]
        if ari_value is not None:
            ari[left, right] = ari[right, left] = float(ari_value)
        if nmi_value is not None:
            nmi[left, right] = nmi[right, left] = float(nmi_value)
        coverage[left, right] = coverage[right, left] = float(metrics["coverage_fraction"])
        records.append({
            "estimator_a": ids[left],
            "estimator_b": ids[right],
            **metrics,
        })

    return AlgorithmAgreementReport(
        estimator_ids=ids,
        ignore_noise=bool(ignore_noise),
        adjusted_rand_matrix=ari,
        normalized_mutual_info_matrix=nmi,
        coverage_matrix=coverage,
        pairwise_records=records,
    )
