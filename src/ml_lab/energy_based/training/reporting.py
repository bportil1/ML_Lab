from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_lab.core.io import ensure_output_dir, write_json
from ml_lab.energy_based.rbm.checkpoint import save as save_rbm

from .base import TrainingResult


def save_result(result: TrainingResult, output_dir: str | Path) -> Path:
    output = ensure_output_dir(output_dir)
    write_json(output / "result.json", result.to_record())
    if result.history:
        pd.DataFrame(result.history).to_csv(output / "training_history.csv", index=False)
    save_rbm(result.model, output / "model.pt")
    if result.generated_samples is not None:
        pd.DataFrame(result.generated_samples, columns=result.feature_names).to_csv(
            output / "generated_samples.csv", index=False
        )
    if result.reference_samples is not None:
        pd.DataFrame(result.reference_samples, columns=result.feature_names).to_csv(
            output / "reference_samples.csv", index=False
        )
    return output
