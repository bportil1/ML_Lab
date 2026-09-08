from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler


def load_csv_features(
    paths: Sequence[str | Path],
    *,
    exclude_columns: Sequence[str] = (),
) -> pd.DataFrame:
    if not paths:
        raise ValueError("at least one CSV path is required")
    frames = [pd.read_csv(Path(path)) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    missing = [column for column in exclude_columns if column not in data.columns]
    if missing:
        raise ValueError(f"excluded columns were not found: {missing}")
    if exclude_columns:
        data = data.drop(columns=list(exclude_columns))
    return validate_matrix(data)[0]


def validate_matrix(X: Any) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    if isinstance(X, pd.DataFrame):
        frame = X.copy()
    else:
        values = np.asarray(X)
        if values.ndim != 2:
            raise ValueError("GAN input must be a two-dimensional feature matrix")
        frame = pd.DataFrame(values, columns=[f"feature_{i}" for i in range(values.shape[1])])
    if frame.shape[0] < 2 or frame.shape[1] < 1:
        raise ValueError("GAN input must contain at least two rows and one feature")
    non_numeric = [column for column in frame.columns if not pd.api.types.is_numeric_dtype(frame[column])]
    if non_numeric:
        raise ValueError(f"GAN input must be numeric; non-numeric columns: {non_numeric}")
    values = frame.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("GAN input must contain only finite values; imputation is not implicit")
    return frame, values, [str(column) for column in frame.columns]


def make_scaler(mode: str):
    if mode == "none":
        return None
    try:
        return {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler(),
        }[mode]
    except KeyError as exc:
        raise ValueError(f"unsupported scaling mode: {mode}") from exc
