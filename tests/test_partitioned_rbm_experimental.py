from __future__ import annotations

import sys

import numpy as np
import pytest

from ml_lab.experimental import get_manifest, run_experiment


def test_partition_ranges_cover_remainders_without_overlap():
    from ml_lab.experimental.partitioned_rbm_training import initial_blocks, merge_blocks

    blocks = initial_blocks(visible_dim=7, hidden_dim=5, partitions=3)
    assert [(b.visible_start, b.visible_end) for b in blocks] == [(0, 3), (3, 5), (5, 7)]
    assert [(b.hidden_start, b.hidden_end) for b in blocks] == [(0, 2), (2, 4), (4, 5)]
    assert sum(b.visible_dim for b in blocks) == 7
    assert sum(b.hidden_dim for b in blocks) == 5

    merged = merge_blocks(blocks, merge_factor=2)
    assert [(b.visible_start, b.visible_end) for b in merged] == [(0, 5), (5, 7)]
    assert [(b.hidden_start, b.hidden_end) for b in merged] == [(0, 4), (4, 5)]
    final = merge_blocks(merged, merge_factor=2)
    assert len(final) == 1
    assert final[0].visible_dim == 7
    assert final[0].hidden_dim == 5


def test_partitioned_experiment_catalog_remains_lazy():
    module = "ml_lab.experimental.partitioned_rbm_training"
    sys.modules.pop(module, None)
    manifest = get_manifest("partitioned_rbm_training")
    assert "partitioned_training" in manifest.capabilities
    assert module not in sys.modules
    description = run_experiment("partitioned_rbm_training")
    assert description["method"] == "hierarchical_feature_blocks_then_merged_refinement"


def test_partitioned_training_uses_feature_axis_and_reaches_full_model(tmp_path):
    pytest.importorskip("torch")
    rng = np.random.default_rng(7)
    X = (rng.random((12, 6)) > 0.5).astype(np.float32)
    result = run_experiment(
        "partitioned_rbm_training",
        mode="fit",
        X=X.tolist(),
        family="bernoulli",
        hidden_dim=4,
        training_config={
            "epochs": 2,
            "batch_size": 4,
            "distribution_monitoring": {"enabled": False},
            "partition_monitoring": {"enabled": False},
        },
        partition_config={
            "initial_partitions": 2,
            "stage_epochs": [1, 1],
        },
        output_dir=str(tmp_path),
    )
    assert result["visible_dim"] == 6
    assert result["hidden_dim"] == 4
    assert result["stage_count"] == 3
    # Two local blocks plus one merged full block. Sample count never enters partition geometry.
    assert [stage["block"]["visible_dim"] for stage in result["stages"]] == [3, 3, 6]
    assert [stage["block"]["hidden_dim"] for stage in result["stages"]] == [2, 2, 4]
    assert (tmp_path / "model.pt").exists()
    assert (tmp_path / "partition_stages.csv").exists()
    assert (tmp_path / "stage_histories" / "level_00_block_000.csv").exists()


def test_partitioned_training_supports_continuous_family():
    torch = pytest.importorskip("torch")
    from ml_lab.experimental.partitioned_rbm_training import fit_partitioned

    rng = np.random.default_rng(3)
    X = rng.normal(size=(10, 4)).astype(np.float32)
    result = fit_partitioned(
        X,
        family="gaussian",
        hidden_dim=4,
        training_config={
            "epochs": 1,
            "batch_size": 5,
            "distribution_monitoring": {"enabled": False},
            "partition_monitoring": {"enabled": False},
        },
        partition_config={"initial_partitions": 2, "stage_epochs": [1, 1]},
    )
    assert result.model.family == "gaussian"
    assert tuple(result.model.W.shape) == (4, 4)
    assert torch.isfinite(result.model.log_sigma).all()


def test_partitioned_training_rejects_more_partitions_than_hidden_units():
    from ml_lab.experimental.partitioned_rbm_training import fit_partitioned

    X = np.ones((8, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="hidden_dim"):
        fit_partitioned(
            X,
            hidden_dim=2,
            partition_config={"initial_partitions": 3},
            training_config={"epochs": 1, "distribution_monitoring": {"enabled": False}},
        )
