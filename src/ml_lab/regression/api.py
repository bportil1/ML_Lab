from __future__ import annotations

from collections.abc import Iterable, Sequence

import pandas as pd

from .config import RegressionSearchConfig
from .data import Target, split_dataset
from .selection import RegressionResult, run_model_suite


def run(
    X: pd.DataFrame,
    y: Target,
    *,
    estimators: Iterable[str] = ("ridge", "random_forest", "extra_trees", "svr", "knn"),
    config: RegressionSearchConfig | None = None,
    test_size: float = 0.2,
    target_names: Sequence[str] | None = None,
) -> list[RegressionResult]:
    config = config or RegressionSearchConfig()
    dataset = split_dataset(
        X,
        y,
        test_size=test_size,
        random_state=config.random_state,
        target_names=target_names,
    )
    return run_model_suite(estimators, dataset, config)
