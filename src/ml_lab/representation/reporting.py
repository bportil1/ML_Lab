from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from ml_lab.core.io import ensure_output_dir, write_json
from .results import RepresentationResult


def save_result(result: RepresentationResult, output_dir: str | Path) -> Path:
    output = ensure_output_dir(output_dir)
    write_json(output / "result.json", result.to_record())

    latent_columns = [f"latent_{index}" for index in range(result.latent.shape[1])]
    pd.DataFrame(result.latent, columns=latent_columns).to_csv(output / "latent.csv", index_label="row_index")
    pd.DataFrame(result.reconstruction, columns=result.feature_names).to_csv(
        output / "reconstruction.csv", index_label="row_index"
    )
    if result.history:
        pd.DataFrame(result.history).to_csv(output / "training_history.csv", index=False)

    if result.method == "pca":
        joblib.dump(result.transformer, output / "transformer.joblib")
    else:
        try:
            import torch
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("saving an autoencoder requires: pip install 'ml-lab[neural]'") from exc
        torch.save(
            {
                "state_dict": result.model.state_dict(),
                "metadata": result.metadata,
                "feature_names": result.feature_names,
            },
            output / "model.pt",
        )
        if result.transformer is not None:
            joblib.dump(result.transformer, output / "scaler.joblib")
    return output
