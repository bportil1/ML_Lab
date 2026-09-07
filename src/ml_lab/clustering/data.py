from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


def load_csv_dataset(
    paths: Sequence[str | Path],
    *,
    label_column: str | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    if not paths:
        raise ValueError("at least one CSV path is required")
    frames = [pd.read_csv(Path(path)) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    y = None
    if label_column is not None:
        if label_column not in data.columns:
            raise ValueError(f"label column {label_column!r} was not found")
        y = data[label_column].copy()
        data = data.drop(columns=[label_column])
    if data.empty:
        raise ValueError("dataset contains no feature columns")
    if data.isna().any().any():
        raise ValueError("feature matrix contains missing values; imputation is not implicit")
    return data, y
