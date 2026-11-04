"""Sprint 6.1: project-local synthetic data and recorded provenance."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from ml_lab import data
from ml_lab.experiments import (
    ConfiguredRunError, ExperimentStore, ExperimentStoreError,
    configure_experiment_model, execute_synthetic_run, generate_synthetic_dataset,
    get_configured_run_status, queue_synthetic_run, request_run_cancellation,
    select_experiment_dataset, verify_selection,
)


def prepared(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    source = project / "measurements.csv"
    source.write_text("a,b,label\n1,2,cat\n2,4,dog\n3,6,cat\n4,8,dog\n5,10,cat\n6,12,dog\n", encoding="utf-8")
    store = ExperimentStore(project)
    experiment = select_experiment_dataset(store, source, target_columns=["label"])
    return store, experiment, source


@pytest.mark.parametrize("method", ["bootstrap", "gaussian_independent", "gaussian_correlated", "student_t"])
def test_statistical_generation_preserves_source_registers_lineage_and_is_selectable(tmp_path, method):
    store, experiment, source = prepared(tmp_path)
    original = source.read_bytes()
    result = generate_synthetic_dataset(
        store, experiment["experiment_id"], {"method": method, "count": 12, "seed": 17},
        expected_revision=experiment["revision"],
    )
    assert result["status"] == "succeeded"
    assert source.read_bytes() == original
    assert store.get_experiment(experiment["experiment_id"])["revision"] == 1
    generated = store.project_root / result["dataset"]["path"]
    parsed = data.read_dataset_frame(generated)
    assert list(parsed) == ["a", "b"]
    assert parsed.shape == (12, 2)
    assert np.isfinite(parsed.to_numpy()).all()
    for item in (result["dataset"], result["provenance"], result["result"]):
        assert store.artifacts.verify(item["artifact_id"])["status"] == "valid"
    event = data.load_provenance_event(generated)
    assert event["source"]["path"] == str(source)
    assert event["derived"]["sha256"] == result["dataset"]["sha256"]
    selected = select_experiment_dataset(store, generated)
    assert verify_selection(store, selected["dataset_ref"])["provenance"]["event_id"] == event["event_id"]
    assert store.get_run(result["run_id"])["status"] == "succeeded"
    assert get_configured_run_status(store, result["run_id"])["step"] == 4


def test_reproducible_same_seed_and_different_seed(tmp_path):
    store, experiment, _ = prepared(tmp_path)
    outputs = []
    for seed in (15, 15, 16):
        run = generate_synthetic_dataset(
            store, experiment["experiment_id"], {"method": "student_t", "count": 100, "seed": seed},
            expected_revision=1,
        )
        outputs.append((store.project_root / run["dataset"]["path"]).read_bytes())
    assert outputs[0] == outputs[1] and outputs[0] != outputs[2]


def test_queued_cancellation_and_stale_revision(tmp_path):
    store, experiment, source = prepared(tmp_path)
    queued = queue_synthetic_run(store, experiment["experiment_id"],
                                  {"method": "bootstrap", "count": 9}, expected_revision=1)
    assert request_run_cancellation(store, queued["run_id"])["acknowledged"]
    with pytest.raises(ExperimentStoreError, match="queued"):
        execute_synthetic_run(store, queued["run_id"])
    store.revise_experiment(experiment["experiment_id"], metadata={"new": True}, expected_revision=1)
    with pytest.raises(ExperimentStoreError, match="revision changed"):
        queue_synthetic_run(store, experiment["experiment_id"],
                            {"method": "bootstrap", "count": 9}, expected_revision=1)
    assert source.is_file()


def test_source_modified_after_queue_fails_closed(tmp_path):
    store, experiment, source = prepared(tmp_path)
    queued = queue_synthetic_run(store, experiment["experiment_id"],
                                  {"method": "bootstrap", "count": 9}, expected_revision=1)
    source.write_text(source.read_text().replace("1,2,cat", "8,2,cat"), encoding="utf-8")
    with pytest.raises(ConfiguredRunError):
        execute_synthetic_run(store, queued["run_id"])
    assert store.get_run(queued["run_id"])["status"] == "failed"
    assert not [art for art in store.artifacts.list_for_run(queued["run_id"]) if art["role"] == "output"]


def test_invalid_methods_counts_and_cross_project_boundaries(tmp_path):
    store, experiment, _ = prepared(tmp_path)
    for plan in ({"method": "tangent_manifold", "count": 10},
                 {"method": "bootstrap", "count": 1},
                 {"method": "bootstrap", "count": 10, "seed": -5},
                 {"method": "student_t", "count": 10, "degrees_of_freedom": 2},
                 {"method": "bootstrap", "count": 10, "output_path": "/tmp/elsewhere"},
                 {"method": "gan", "count": 10},
                 {"method": "rbm", "count": 10}):
        with pytest.raises(ExperimentStoreError):
            queue_synthetic_run(store, experiment["experiment_id"], plan, expected_revision=1)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(ExperimentStoreError):
        queue_synthetic_run(ExperimentStore(other), experiment["experiment_id"],
                            {"method": "bootstrap", "count": 10}, expected_revision=1)


def test_model_sampling_uses_existing_gan_engine_without_checkpoint_reuse(tmp_path):
    pytest.importorskip("torch")
    store, experiment, _ = prepared(tmp_path)
    # GAN requires an unsupervised feature-only selection.
    experiment = select_experiment_dataset(store, store.project_root / "measurements.csv",
                                           experiment_id=experiment["experiment_id"],
                                           feature_columns=["a", "b"], expected_revision=1)
    configured = configure_experiment_model(store, experiment["experiment_id"],
                 family="gan", method="gan",
                 settings={"model_config": {"generator_hidden_dims": [8], "discriminator_hidden_dims": [8],
                                             "latent_dim": 3},
                           "training_config": {"epochs": 1, "batch_size": 3}},
                 expected_revision=experiment["revision"])
    run = generate_synthetic_dataset(store, configured["experiment_id"],
                                     {"method": "gan", "count": 7, "seed": 25},
                                     expected_revision=configured["revision"])
    assert data.read_dataset_frame(store.project_root / run["dataset"]["path"]).shape == (7, 2)
    assert store.get_run(run["run_id"])["environment"]["sampling"] == "train-now"


def test_model_sampling_uses_existing_rbm_gibbs_chain(tmp_path):
    pytest.importorskip("torch")
    store, experiment, _ = prepared(tmp_path)
    experiment = select_experiment_dataset(store, store.project_root / "measurements.csv",
                                           experiment_id=experiment["experiment_id"],
                                           feature_columns=["a", "b"], expected_revision=1)
    configured = configure_experiment_model(store, experiment["experiment_id"],
                 family="rbm", method="gaussian",
                 settings={"hidden_dim": 3, "training_config": {"epochs": 1, "batch_size": 3}},
                 expected_revision=experiment["revision"])
    run = generate_synthetic_dataset(store, configured["experiment_id"],
                                     {"method": "rbm", "count": 7, "seed": 25},
                                     expected_revision=configured["revision"])
    assert data.read_dataset_frame(store.project_root / run["dataset"]["path"]).shape == (7, 2)


def test_output_tampering_invalidates_artifact_and_selection(tmp_path):
    store, experiment, _ = prepared(tmp_path)
    run = generate_synthetic_dataset(store, experiment["experiment_id"],
                                     {"method": "bootstrap", "count": 8}, expected_revision=1)
    generated = store.project_root / run["dataset"]["path"]
    generated.write_text(generated.read_text().replace("1.0", "99.0"), encoding="utf-8")
    assert store.artifacts.verify(run["dataset"]["artifact_id"])["status"] != "valid"
    from ml_lab.experiments import DatasetSelectionError, describe_selection
    with pytest.raises(DatasetSelectionError):
        describe_selection(store, generated)


def test_running_cancellation_request_is_acknowledged_at_safe_stage(tmp_path, monkeypatch):
    store, experiment, _ = prepared(tmp_path)
    queued = queue_synthetic_run(store, experiment["experiment_id"],
                                  {"method": "bootstrap", "count": 8}, expected_revision=1)
    from ml_lab.experiments import synthetic
    generator = synthetic._statistical_samples

    def interrupted(values, plan):
        assert not request_run_cancellation(store, queued["run_id"])["acknowledged"]
        return generator(values, plan)

    monkeypatch.setattr(synthetic, "_statistical_samples", interrupted)
    with pytest.raises(ConfiguredRunError):
        execute_synthetic_run(store, queued["run_id"])
    assert store.get_run(queued["run_id"])["status"] == "cancelled"
    assert not [row for row in store.artifacts.list_for_run(queued["run_id"])
                if row["role"] == "output"]
