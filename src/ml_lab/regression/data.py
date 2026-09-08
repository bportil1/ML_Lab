from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import pandas as pd
from sklearn.model_selection import train_test_split


Target = pd.Series | pd.DataFrame


@dataclass(slots=True)
class RegressionDataset:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: Target
    y_test: Target
    feature_names: list[str]
    target_names: list[str]

    @property
    def multi_output(self) -> bool:
        return isinstance(self.y_train, pd.DataFrame) and self.y_train.shape[1] > 1


def _normalize_target_columns(target_columns: str | Sequence[str]) -> list[str]:
    if isinstance(target_columns, str):
        columns = [target_columns]
    else:
        columns = [str(column) for column in target_columns]
    if not columns:
        raise ValueError("at least one target column is required")
    if len(set(columns)) != len(columns):
        raise ValueError("target columns must be unique")
    return columns


def load_csv_dataset(
    paths: Sequence[str | Path],
    target_columns: str | Sequence[str] = "target",
) -> tuple[pd.DataFrame, Target]:
    if not paths:
        raise ValueError("at least one CSV path is required")
    frames = [pd.read_csv(Path(path)) for path in paths]
    data = pd.concat(frames, ignore_index=True)
    targets = _normalize_target_columns(target_columns)
    missing = [column for column in targets if column not in data.columns]
    if missing:
        raise ValueError(f"target column(s) not found: {', '.join(missing)}")

    X = data.drop(columns=targets).copy()
    y_frame = data[targets].copy()
    if X.empty:
        raise ValueError("dataset contains no feature columns")
    if X.isna().any().any():
        raise ValueError("feature matrix contains missing values; imputation is not implicit")
    if y_frame.isna().any().any():
        raise ValueError("target columns contain missing values")

    try:
        y_frame = y_frame.astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError("regression target columns must be numeric") from exc

    y: Target = y_frame.iloc[:, 0].rename(targets[0]) if len(targets) == 1 else y_frame
    return X, y


def split_dataset(
    X: pd.DataFrame,
    y: Target,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    target_names: Sequence[str] | None = None,
) -> RegressionDataset:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of rows")
    if len(X) < 3:
        raise ValueError("regression requires at least three rows")

    if isinstance(y, pd.DataFrame):
        names = [str(column) for column in y.columns]
    else:
        names = [str(y.name or "target")]
    if target_names is not None:
        supplied = [str(name) for name in target_names]
        if len(supplied) != len(names):
            raise ValueError("target_names length must match the number of targets")
        names = supplied

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )
    return RegressionDataset(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.reset_index(drop=True),
        y_test=y_test.reset_index(drop=True),
        feature_names=[str(column) for column in X.columns],
        target_names=names,
    )
