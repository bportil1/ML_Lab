from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


@dataclass(slots=True)
class ClassificationDataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_names: list[str]
    label_name: str

    @property
    def classes(self) -> np.ndarray:
        return np.unique(self.y_train)

    @property
    def task_type(self) -> str:
        return "binary" if len(self.classes) == 2 else "multiclass"


def load_csv_dataset(paths: Sequence[str | Path], label_column: str = "label") -> tuple[pd.DataFrame, pd.Series]:
    if not paths:
        raise ValueError("at least one CSV path is required")
    frames = [pd.read_csv(Path(path)) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    if label_column not in data.columns:
        raise ValueError(f"label column {label_column!r} was not found")
    y = data[label_column].copy()
    X = data.drop(columns=[label_column]).copy()
    if X.empty:
        raise ValueError("dataset contains no feature columns")
    if y.isna().any():
        raise ValueError("label column contains missing values")
    if X.isna().any().any():
        raise ValueError("feature matrix contains missing values; imputation is not implicit")
    if y.nunique() < 2:
        raise ValueError("classification requires at least two classes")
    return X, y


def split_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    label_name: str = "label",
) -> ClassificationDataset:
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return ClassificationDataset(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        feature_names=[str(c) for c in X.columns],
        label_name=label_name,
    )
