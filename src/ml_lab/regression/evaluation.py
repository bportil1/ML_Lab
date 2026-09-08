from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import (
    explained_variance_score,
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)


def regression_metrics(model: Any, X_test: Any, y_test: Any) -> dict[str, float]:
    y_pred = model.predict(X_test)
    mse = float(mean_squared_error(y_test, y_pred))
    return {
        "rmse": float(np.sqrt(mse)),
        "mae": float(mean_absolute_error(y_test, y_pred)),
        "r2": float(r2_score(y_test, y_pred, multioutput="uniform_average")),
        "median_absolute_error": float(
            median_absolute_error(y_test, y_pred, multioutput="uniform_average")
        ),
        "explained_variance": float(
            explained_variance_score(y_test, y_pred, multioutput="uniform_average")
        ),
    }
