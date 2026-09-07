from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    fbeta_score,
    hamming_loss,
    jaccard_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    zero_one_loss,
)


def _score_values(model: Any, X: Any) -> np.ndarray | None:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(X))
    if hasattr(model, "decision_function"):
        return np.asarray(model.decision_function(X))
    return None


def classification_metrics(model: Any, X_test: Any, y_test: Any) -> dict[str, float | None]:
    y_pred = model.predict(X_test)
    classes = np.unique(y_test)
    binary = len(classes) == 2
    average = "binary" if binary else "weighted"
    kwargs: dict[str, Any] = {"average": average, "zero_division": 0}
    if binary:
        kwargs["pos_label"] = model.classes_[-1]

    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, **kwargs)),
        "recall": float(recall_score(y_test, y_pred, **kwargs)),
        "f1": float(f1_score(y_test, y_pred, **kwargs)),
        "f2": float(fbeta_score(y_test, y_pred, beta=2, **kwargs)),
        "matthews_corrcoef": float(matthews_corrcoef(y_test, y_pred)),
        "cohen_kappa": float(cohen_kappa_score(y_test, y_pred)),
        "jaccard": float(jaccard_score(y_test, y_pred, **kwargs)),
        "hamming_loss": float(hamming_loss(y_test, y_pred)),
        "zero_one_loss": float(zero_one_loss(y_test, y_pred)),
        "roc_auc": None,
    }
    scores = _score_values(model, X_test)
    if scores is not None:
        try:
            if binary:
                if scores.ndim == 2:
                    scores = scores[:, 1]
                metrics["roc_auc"] = float(roc_auc_score(y_test, scores))
            elif scores.ndim == 2 and scores.shape[1] == len(classes):
                metrics["roc_auc"] = float(
                    roc_auc_score(y_test, scores, multi_class="ovr", average="weighted")
                )
        except ValueError:
            metrics["roc_auc"] = None
    return metrics
