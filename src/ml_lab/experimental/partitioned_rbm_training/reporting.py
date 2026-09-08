from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml_lab.core.io import ensure_output_dir, write_json
from ml_lab.energy_based.rbm.checkpoint import save as save_rbm

from .training import PartitionedRBMResult


def save_result(result: PartitionedRBMResult, output_dir: str | Path) -> Path:
    output = ensure_output_dir(output_dir)
    write_json(output / "result.json", result.to_record())
    save_rbm(result.model, output / "model.pt")

    stage_records = []
    histories = output / "stage_histories"
    histories.mkdir(exist_ok=True)
    for stage_index, stage in enumerate(result.stages):
        record = stage.to_record()
        stage_records.append({
            "stage_index": stage_index,
            "level": record["level"],
            "block_index": record["block_index"],
            "visible_start": stage.block.visible_start,
            "visible_end": stage.block.visible_end,
            "hidden_start": stage.block.hidden_start,
            "hidden_end": stage.block.hidden_end,
            "epochs_requested": record["epochs_requested"],
            "epochs_completed": record["epochs_completed"],
            "stop_reason": record["stop_reason"],
            **{f"metric_{key}": value for key, value in record["metrics"].items()},
        })
        if stage.result.history:
            pd.DataFrame(stage.result.history).to_csv(
                histories / f"level_{stage.level:02d}_block_{stage.block_index:03d}.csv",
                index=False,
            )
    pd.DataFrame(stage_records).to_csv(output / "partition_stages.csv", index=False)
    return output
