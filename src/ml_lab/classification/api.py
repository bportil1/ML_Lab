from __future__ import annotations

from collections.abc import Iterable

import pandas as pd

from .config import ClassificationSearchConfig
from .data import split_dataset
from .selection import ClassificationResult, run_model_suite


def run(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    estimators: Iterable[str] = ("rf", "hgbc", "lda", "qda", "ridge"),
    config: ClassificationSearchConfig | None = None,
    test_size: float = 0.2,
    label_name: str = "label",
) -> list[ClassificationResult]:
    config = config or ClassificationSearchConfig()
    dataset = split_dataset(
        X,
        y,
        test_size=test_size,
        random_state=config.random_state,
        label_name=label_name,
    )
    return run_model_suite(estimators, dataset, config)
