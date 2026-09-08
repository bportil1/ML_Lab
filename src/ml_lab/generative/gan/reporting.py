from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from ml_lab.core.io import ensure_output_dir, write_json
from ml_lab.neural.backend import require_torch

from .results import GANResult


def save_result(result: GANResult, output_dir: str | Path) -> Path:
    output = ensure_output_dir(output_dir)
    write_json(output / "result.json", result.to_record())
    pd.DataFrame(result.generated_samples, columns=result.feature_names).to_csv(
        output / "generated_samples.csv", index_label="row_index"
    )
    if result.history:
        pd.DataFrame(result.history).to_csv(output / "training_history.csv", index=False)

    torch = require_torch(purpose="GAN model persistence")
    torch.save(
        {
            "state_dict": result.generator.state_dict(),
            "metadata": result.metadata,
            "feature_names": result.feature_names,
        },
        output / "generator.pt",
    )
    torch.save(
        {
            "state_dict": result.discriminator.state_dict(),
            "metadata": result.metadata,
            "feature_names": result.feature_names,
        },
        output / "discriminator.pt",
    )
    if result.transformer is not None:
        joblib.dump(result.transformer, output / "scaler.joblib")
    return output
