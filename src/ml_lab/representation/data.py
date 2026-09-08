from __future__ import annotations

from pathlib import Path
from typing import Sequence

import pandas as pd


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
    if data.empty:
        raise ValueError("dataset contains no feature columns")
    non_numeric = [column for column in data.columns if not pd.api.types.is_numeric_dtype(data[column])]
    if non_numeric:
        raise ValueError(f"representation input must be numeric; non-numeric columns: {non_numeric}")
    if data.isna().any().any():
        raise ValueError("feature matrix contains missing values; imputation is not implicit")
    return data
