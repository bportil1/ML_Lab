"""S3.1: stable, typed model methods and revision-safe dataset-to-model handoff."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import csv
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ml_lab.experiments import (
    DatasetSelectionError,
    ExperimentStore,
    ModelConfigurationError,
    configure_experiment_model,
    describe_model_method,
    list_model_families,
    list_model_methods,
    prepare_configured_model_handoff,
    select_experiment_dataset,
    validate_model_configuration,
)


def _experiment(tmp_path: Path, *, targets: bool = False):
    project = tmp_path / "project"
    project.mkdir()
    source = project / "source.csv"
    with source.open("w", encoding="utf-8", newline="") as out:
        writer = csv.writer(out)
        writer.writerow(["x", "z", "label"])
        writer.writerows([[n, n * 2, "odd" if n % 2 else "even"] for n in range(1, 11)])
    store = ExperimentStore(project)
    experiment = select_experiment_dataset(
        store, source, feature_columns=["x", "z"], target_columns=["label"] if targets else [],
    )
    return store, experiment, source


def test_all_stable_families_and_methods_are_read_only_pure_json():
    expected = {"classification", "regression", "clustering", "representation", "rbm", "gan"}
    assert {item["id"] for item in list_model_families()} == expected
    assert {item["id"] for item in list_model_methods("representation")} == {
        "pca", "mlp_autoencoder", "transformer_autoencoder",
    }
    assert {item["id"] for item in list_model_methods("rbm")} == {
        "bernoulli", "gaussian", "student_t_poe",
    }
    assert {item["id"] for item in list_model_methods("gan")} == {"gan"}
    assert {item["id"] for item in list_model_methods("classification")} >= {"ridge", "rf"}
    assert {item["id"] for item in list_model_methods("regression")} >= {"ridge", "random_forest"}
    assert {item["id"] for item in list_model_methods("clustering")} >= {"kmeans", "dbscan"}
    for family in expected:
        for method in list_model_methods(family):
            details = describe_model_method(family, method["id"])
            assert details["task"] == family
            assert details["schema"] == "ml-lab.model-configuration@1"
            json.dumps(details, allow_nan=False)
            assert isinstance(details["controls"], dict)
            requested = {"estimators": [next(m["id"] for m in list_model_methods(family) if m["id"] != "suite")]} if method["id"] == "suite" else None
            settings = validate_model_configuration(family, method["id"], requested)
            json.dumps(settings, allow_nan=False)
    assert "generated_sample_count" in {f["name"] for f in describe_model_method("gan", "gan")["controls"]["training_config"]}
    assert "model_selection" in {f["name"] for f in describe_model_method("rbm", "gaussian")["controls"]["training_config"]}


def test_catalog_import_does_not_force_optional_torch_or_pah_packages():
    script = """import sys
from ml_lab.experiments import list_model_families, list_model_methods, describe_model_method
for x in list_model_families():
 for method in list_model_methods(x['id']):
  describe_model_method(x['id'],method['id'])
assert 'torch' not in sys.modules
assert 'flask' not in sys.modules
assert 'pah' not in sys.modules
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("family,method,settings,payload_piece", [
    ("classification", "ridge", {"config": {"cv_folds": 3, "refit": False}, "test_size": 0.25}, {"estimators": ["ridge"]}),
    ("regression", "ridge", {"config": {"cv_folds": 4}}, {"estimators": ["ridge"]}),
    ("clustering", "kmeans", {"config": {"repeats": 2}}, {"estimators": ["kmeans"]}),
    ("classification", "suite", {"estimators": ["ridge", "rf"]}, {"estimators": ["ridge", "rf"]}),
    ("regression", "suite", {"estimators": ["ridge", "random_forest"]}, {"estimators": ["ridge", "random_forest"]}),
    ("clustering", "suite", {"estimators": ["kmeans", "dbscan"]}, {"estimators": ["kmeans", "dbscan"]}),
    ("representation", "pca", {"config": {"n_components": 1}}, {"method": "pca"}),
    ("representation", "mlp_autoencoder", {"model_config": {"hidden_dims": [12, 4], "latent_dim": 2}, "training_config": {"epochs": 3}}, {"method": "mlp_autoencoder"}),
    ("representation", "transformer_autoencoder", {"model_config": {"model_dim": 8, "nhead": 2}, "training_config": {"batch_size": 4}}, {"method": "transformer_autoencoder"}),
    ("gan", "gan", {"model_config": {"generator_type": "mlp", "discriminator_type": "transformer"}, "training_config": {"generator_steps": 2}}, {}),
    ("rbm", "gaussian", {"hidden_dim": 4, "model_config": {"settings": {"initial_log_sigma": 0.3}}, "training_config": {"gibbs_steps": 3, "model_selection": {"enabled": True}}}, {"family": "gaussian", "hidden_dim": 4}),
])
def test_valid_configs_are_legacy_compatible(family, method, settings, payload_piece):
    config = validate_model_configuration(family, method, settings, n_features=2)
    for key, value in payload_piece.items():
        assert config[key] == value
    assert json.loads(json.dumps(config, allow_nan=False)) == config


@pytest.mark.parametrize("family,method,settings", [
    ("classification", "no_such_estimator", {}),
    ("representation", "no_such_method", {}),
    ("rbm", "unknown", {}),
    ("gan", "gan", {"model_config": {"generator_type": "invalid"}}),
    ("gan", "gan", {"training_config": {"epochs": 0}}),
    ("classification", "ridge", {"test_size": 1.0}),
    ("classification", "ridge", {"test_size": "0.2"}),
    ("classification", "ridge", {"test_size": True}),
    ("classification", "ridge", {"config": {"cv_folds": 1}}),
    ("classification", "ridge", {"config": {"cv_folds": False}}),
    ("regression", "ridge", {"config": {"no_such_setting": 0}}),
    ("clustering", "kmeans", {"config": {"repeats": 0}}),
    ("representation", "pca", {"config": {"n_components": 0}}),
    ("representation", "mlp_autoencoder", {"model_config": {"hidden_dims": [0]}}),
    ("representation", "transformer_autoencoder", {"model_config": {"model_dim": 5, "nhead": 2}}),
    ("rbm", "gaussian", {"hidden_dim": 0}),
    ("rbm", "gaussian", {"model_config": {"settings": {"unknown": 1}}}),
    ("rbm", "bernoulli", {"training_config": {"model_selection": {"patience": -1}}}),
    ("gan", "gan", {"X": [[0]]}),
    ("gan", "gan", {"training_config": {"checkpoint_path": "/tmp/unsafe.pt"}}),
    ("representation", "mlp_autoencoder", {"training_config": {"checkpoint_path": "../weights.pt"}}),
    ("classification", "suite", {}),
    ("classification", "suite", {"estimators": []}),
    ("classification", "suite", {"estimators": ["ridge", "ridge"]}),
    ("classification", "suite", {"estimators": ["ridge", "not-an-estimator"]}),
    ("regression", "ridge", {"estimators": ["random_forest"]}),
])
def test_invalid_config_is_rejected_before_persisting(family, method, settings):
    with pytest.raises(ModelConfigurationError):
        validate_model_configuration(family, method, settings)


def test_empty_or_wrong_settings_types_are_rejected():
    for settings in ([], "", 42, False):
        with pytest.raises(ModelConfigurationError):
            validate_model_configuration("representation", "pca", settings)
    with pytest.raises(ModelConfigurationError):
        validate_model_configuration("representation", "pca", {"config": []})
    with pytest.raises(ModelConfigurationError):
        validate_model_configuration("regression", "ridge", {"config": {"error_score": float("nan")}})


def test_selection_to_model_to_payload_and_revision_idempotency(tmp_path):
    store, selected, source = _experiment(tmp_path, targets=True)
    configured = configure_experiment_model(store, selected["experiment_id"], family="classification", method="ridge",
                                            settings={"test_size": 0.3, "config": {"cv_folds": 2}}, expected_revision=1)
    assert configured["revision"] == 2
    assert configured["dataset_ref"] == selected["dataset_ref"]
    assert configured["model_spec"]["method"] == "ridge"
    again = configure_experiment_model(store, selected["experiment_id"], family="classification", method="ridge",
                                       settings={"test_size": 0.3, "config": {"cv_folds": 2}}, expected_revision=2)
    assert again["revision"] == 2
    handoff = prepare_configured_model_handoff(store, selected["experiment_id"])
    assert handoff["revision"] == 2
    assert handoff["task"] == "classification"
    assert handoff["payload"]["estimators"] == ["ridge"]
    assert handoff["payload"]["test_size"] == 0.3
    assert handoff["payload"]["y"][:2] == ["odd", "even"]
    assert handoff["payload"]["X"][:2] == [[1.0, 2.0], [2.0, 4.0]]
    assert store.get_experiment(selected["experiment_id"], revision=1)["model_spec"] == {}
    changed = configure_experiment_model(store, selected["experiment_id"], family="classification", method="rf", expected_revision=2)
    assert changed["revision"] == 3
    old = prepare_configured_model_handoff(store, selected["experiment_id"], revision=2)
    assert old["payload"]["estimators"] == ["ridge"]
    assert prepare_configured_model_handoff(store, selected["experiment_id"])["payload"]["estimators"] == ["rf"]
    assert source.exists()


def test_data_change_resets_model_configuration_not_history(tmp_path):
    store, selected, source = _experiment(tmp_path)
    configured = configure_experiment_model(store, selected["experiment_id"], family="representation", method="pca", expected_revision=1)
    changed = select_experiment_dataset(store, source, feature_columns=["z"],
                                        experiment_id=selected["experiment_id"], expected_revision=2)
    assert changed["revision"] == 3 and changed["model_spec"] == {}
    with pytest.raises(ModelConfigurationError, match="does not have a configured"):
        prepare_configured_model_handoff(store, selected["experiment_id"])
    assert prepare_configured_model_handoff(store, selected["experiment_id"], revision=2)["payload"]["feature_names"] == ["x", "z"]
    assert configured["model_spec"]["method"] == "pca"


def test_target_compatibility_is_checked_before_writing(tmp_path):
    store, selected, _ = _experiment(tmp_path, targets=True)
    with pytest.raises(ModelConfigurationError, match="requires"):
        configure_experiment_model(store, selected["experiment_id"], family="gan", method="gan", expected_revision=1)
    with pytest.raises(ModelConfigurationError, match="requires"):
        configure_experiment_model(store, selected["experiment_id"], family="representation", method="pca", expected_revision=1)
    assert store.get_experiment(selected["experiment_id"])["revision"] == 1
    cluster = configure_experiment_model(store, selected["experiment_id"], family="clustering", method="kmeans", expected_revision=1)
    assert prepare_configured_model_handoff(store, selected["experiment_id"])["payload"]["y_true"][:2] == ["odd", "even"]
    assert cluster["revision"] == 2
    with pytest.raises(ModelConfigurationError):
        configure_experiment_model(store, selected["experiment_id"], family="classification", method="ridge", expected_revision=1)


def test_updated_source_is_stale_and_cannot_publish_model(tmp_path):
    store, selected, source = _experiment(tmp_path)
    source.write_text("x,z,label\n100,200,changed\n", encoding="utf-8")
    with pytest.raises(DatasetSelectionError):
        configure_experiment_model(store, selected["experiment_id"], family="representation", method="pca", expected_revision=1)
    assert store.get_experiment(selected["experiment_id"])["revision"] == 1


def test_concurrent_writers_only_publish_one_model_revision(tmp_path):
    store, selected, _ = _experiment(tmp_path)
    def set_model(method):
        return configure_experiment_model(store, selected["experiment_id"], family="representation",
                                          method=method, expected_revision=1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = [pool.submit(set_model, method) for method in ("pca", "mlp_autoencoder")]
        ok, blocked = [], []
        for item in results:
            try:
                ok.append(item.result())
            except ModelConfigurationError as exc:
                blocked.append(str(exc))
    assert len(ok) == 1 and len(blocked) == 1
    assert store.get_experiment(selected["experiment_id"])["revision"] == 2


def test_legacy_recorded_experiments_cannot_be_reconfigured(tmp_path):
    store, selected, _ = _experiment(tmp_path)
    record = store.create_experiment("old", dataset_ref=selected["dataset_ref"],
                                     metadata={"adapter": "ml-lab.recorded-legacy-task@1"})
    with pytest.raises(ModelConfigurationError, match="Legacy-recorded"):
        configure_experiment_model(store, record["experiment_id"], family="representation", method="pca", expected_revision=1)
    with pytest.raises(ModelConfigurationError, match="Select a Data Lab dataset"):
        configure_experiment_model(store, store.create_experiment("blank")["experiment_id"], family="gan", method="gan", expected_revision=1)


def test_prepared_model_payload_executes_through_unchanged_dispatcher(tmp_path):
    from ml_lab.application import execute_task
    store, selected, _ = _experiment(tmp_path)
    configured = configure_experiment_model(store, selected["experiment_id"], family="representation", method="pca",
                                            settings={"config": {"n_components": 1}}, expected_revision=1)
    handoff = prepare_configured_model_handoff(store, configured["experiment_id"])
    output = execute_task(handoff["task"], handoff["payload"])
    assert output["task"] == "representation"
    assert len(output["result"]["latent"]) == 10
    assert store.get_experiment(configured["experiment_id"])["revision"] == 2  # no implicit run
    assert store.list_runs(configured["experiment_id"]) == []


def test_invalid_persisted_model_configuration_cannot_be_materialized(tmp_path):
    store, selected, _ = _experiment(tmp_path)
    configured = configure_experiment_model(store, selected["experiment_id"], family="rbm",
                                            method="bernoulli", expected_revision=1)
    broken = dict(configured["model_spec"])
    broken["settings"] = dict(broken["settings"], family="gaussian")
    store.revise_experiment(selected["experiment_id"], dataset_ref=selected["dataset_ref"],
                            model_spec=broken, expected_revision=2)
    with pytest.raises(ModelConfigurationError, match="not canonical"):
        prepare_configured_model_handoff(store, selected["experiment_id"])


def test_cross_project_model_revision_cannot_reach_other_project(tmp_path):
    store, selected, _ = _experiment(tmp_path)
    other_root = tmp_path / "other"
    other_root.mkdir()
    other_store = ExperimentStore(other_root)
    from ml_lab.experiments import ExperimentNotFound
    with pytest.raises(ExperimentNotFound):
        configure_experiment_model(other_store, selected["experiment_id"],
                                   family="representation", method="pca", expected_revision=1)
    assert other_store.list_experiments() == []


def test_all_family_defaults_can_roundtrip_as_persisted_payloads(tmp_path):
    for family, method, targets in [
        ("classification", "ridge", True), ("regression", "ridge", True),
        ("clustering", "kmeans", True), ("representation", "pca", False),
        ("rbm", "student_t_poe", False), ("gan", "gan", False),
    ]:
        root = tmp_path / family
        root.mkdir()
        source = root / "data.csv"
        source.write_text("x,y,label\n1,2,5\n3,4,9\n", encoding="utf-8")
        store = ExperimentStore(root)
        selected = select_experiment_dataset(store, source,
                                             feature_columns=["x", "y"], target_columns=["label"] if targets else [])
        configured = configure_experiment_model(store, selected["experiment_id"],
                                                family=family, method=method, expected_revision=1)
        handoff = prepare_configured_model_handoff(store, configured["experiment_id"])
        assert handoff["payload"]["X"] == [[1.0, 2.0], [3.0, 4.0]]
        assert handoff["task"] == family
        assert handoff["revision"] == 2
