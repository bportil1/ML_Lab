from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class BatchResolution:
    sample_count: int
    requested_batch_size: int
    effective_batch_size: int
    requested_drop_last: bool
    effective_drop_last: bool
    batch_count: int

    def to_record(self) -> dict[str, int | bool]:
        return {
            "sample_count": self.sample_count,
            "requested_batch_size": self.requested_batch_size,
            "effective_batch_size": self.effective_batch_size,
            "requested_drop_last": self.requested_drop_last,
            "effective_drop_last": self.effective_drop_last,
            "batch_count": self.batch_count,
        }



def load_csv_features(paths: Sequence[str | Path], *, exclude_columns: Sequence[str] = ()) -> pd.DataFrame:
    if not paths:
        raise ValueError("at least one CSV path is required")
    frames = [pd.read_csv(Path(path)) for path in paths]
    frame = pd.concat(frames, ignore_index=True)
    missing = [column for column in exclude_columns if column not in frame.columns]
    if missing:
        raise ValueError(f"excluded columns were not found: {missing}")
    if exclude_columns:
        frame = frame.drop(columns=list(exclude_columns))
    validate_matrix(frame)
    return frame

def validate_matrix(X: Any) -> tuple[np.ndarray, list[str]]:
    if hasattr(X, "detach") and hasattr(X, "cpu"):
        X = X.detach().cpu().numpy()
    if isinstance(X, pd.DataFrame):
        frame = X.copy()
        names = [str(column) for column in frame.columns]
        non_numeric = [column for column in frame.columns if not pd.api.types.is_numeric_dtype(frame[column])]
        if non_numeric:
            raise ValueError(f"RBM input must be numeric; non-numeric columns: {non_numeric}")
        values = frame.to_numpy(dtype=np.float32)
    else:
        values = np.asarray(X, dtype=np.float32)
        if values.ndim != 2:
            raise ValueError("RBM input must be a two-dimensional feature matrix")
        names = [f"feature_{index}" for index in range(values.shape[1])]
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise ValueError("RBM input must contain at least one row and one feature")
    if not np.isfinite(values).all():
        raise ValueError("RBM input must contain only finite values; imputation is not implicit")
    return values, names


def resolve_batching(sample_count: int, batch_size: int, drop_last: bool) -> BatchResolution:
    if sample_count < 1:
        raise ValueError("sample_count must be >= 1")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")
    effective_batch = min(int(batch_size), int(sample_count))
    effective_drop = bool(drop_last and sample_count >= effective_batch)
    if effective_drop:
        count = sample_count // effective_batch
    else:
        count = (sample_count + effective_batch - 1) // effective_batch
    if count < 1:
        effective_drop = False
        count = 1
    return BatchResolution(
        sample_count=int(sample_count),
        requested_batch_size=int(batch_size),
        effective_batch_size=int(effective_batch),
        requested_drop_last=bool(drop_last),
        effective_drop_last=effective_drop,
        batch_count=int(count),
    )


def iter_batches(
    tensor,
    resolution: BatchResolution,
    *,
    shuffle: bool,
    generator,
) -> Iterator[Any]:
    torch = __import__("torch")
    if shuffle:
        indices = torch.randperm(len(tensor), generator=generator)
    else:
        indices = torch.arange(len(tensor))
    step = resolution.effective_batch_size
    for start in range(0, len(indices), step):
        batch_indices = indices[start : start + step]
        if resolution.effective_drop_last and len(batch_indices) < step:
            continue
        yield tensor[batch_indices]
