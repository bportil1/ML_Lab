"""Sprint 1.3: opt-in legacy task recording and provenance integration."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
import pytest

from ml_lab.application import PayloadError, execute_task
from ml_lab.experiments import (
    ExperimentStore, ExperimentStoreError, RecordedTaskError, execute_recorded_task,
)


def test_recorded_application_response_keeps_legacy_envelope_and_sources(tmp_path: Path):
    source = tmp_path / "dataset.csv"
    source.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    payload = {"paths": [str(source)], "preview_rows": 2}
    legacy = execute_task("data.inspect", payload)
    recorded = execute_recorded_task("data.inspect", payload, project_root=tmp_path)
    assert recorded["response"] == legacy
    assert recorded["status"] == "succeeded"
    store = ExperimentStore(tmp_path)
    run = store.get_run(recorded["run_id"])
    assert run["status"] == "succeeded"
    experiment = store.get_experiment(recorded["experiment_id"])
    assert experiment["model_spec"]["task"] == "data.inspect"
    assert experiment["dataset_ref"]["sha256"] == experiment["metadata"]["payload_sha256"]
    assert experiment["dataset_ref"]["source_files"][0]["path"] == "dataset.csv"
    assert experiment["dataset_ref"]["source_files"][0]["sha256"]
    assert "X" not in experiment["model_spec"]["settings"]
    assert {a["kind"] for a in recorded["artifacts"]} == {"dataset", "payload", "result"}
    for artifact in recorded["artifacts"]:
        assert artifact["experiment_id"] == recorded["experiment_id"]
        assert store.artifacts.verify(artifact["artifact_id"])["status"] == "valid"
    output = next(a for a in recorded["artifacts"] if a["kind"] == "result")
    assert len(output["source_artifact_ids"]) == 2
    assert json.loads((tmp_path / output["path"]).read_text()) == legacy
    input_artifact = next(a for a in recorded["artifacts"] if a["kind"] == "payload")
    assert json.loads((tmp_path / input_artifact["path"]).read_text()) == payload


def test_same_path_changed_file_must_not_reuse_old_revision(tmp_path: Path):
    source = tmp_path / "data.csv"
    source.write_text("a\n1\n", encoding="utf-8")
    payload = {"paths": [str(source)]}
    previous = execute_recorded_task("data.inspect", payload, project_root=tmp_path)
    source.write_text("a\n9\n", encoding="utf-8")
    with pytest.raises(ExperimentStoreError, match="does not match"):
        execute_recorded_task("data.inspect", payload, project_root=tmp_path,
                              experiment_id=previous["experiment_id"])
    assert len(ExperimentStore(tmp_path).list_runs(previous["experiment_id"])) == 1
    new = execute_recorded_task("data.inspect", payload, project_root=tmp_path)
    assert new["experiment_id"] != previous["experiment_id"]
    assert (ExperimentStore(tmp_path).get_experiment(new["experiment_id"])
            ["dataset_ref"]["source_files"][0]["sha256"] !=
            ExperimentStore(tmp_path).get_experiment(previous["experiment_id"])
            ["dataset_ref"]["source_files"][0]["sha256"])


def test_numpy_payload_and_in_memory_model_not_in_sqlite(tmp_path: Path):
    X = np.array([[0.1, 0.2], [0.2, 0.5], [0.3, 0.7], [0.6, 0.8]], dtype=np.float64)
    payload = {"X": X, "method": "pca", "config": {"n_components": 1}}
    response = execute_recorded_task("representation", payload, project_root=tmp_path, seed=12)
    assert response["response"]["task"] == "representation"
    assert "latent" in response["response"]["result"]
    store = ExperimentStore(tmp_path)
    run = store.get_run(response["run_id"])
    assert run["seed"] == 12
    experiment = store.get_experiment(response["experiment_id"])
    assert "X" not in experiment["model_spec"]["settings"]
    assert experiment["dataset_ref"]["row_count"] == 4
    input_artifact = next(a for a in response["artifacts"] if a["kind"] == "payload")
    assert json.loads((tmp_path / input_artifact["path"]).read_text())["X"] == X.tolist()
    assert not list(tmp_path.rglob("*.pt"))  # adapter invents no checkpoint


def test_matching_repeated_invocation_and_pinned_revision(tmp_path: Path):
    source = tmp_path / "x.csv"
    source.write_text("x\n1\n", encoding="utf-8")
    payload = {"paths": [str(source)]}
    first = execute_recorded_task("data.inspect", payload, project_root=tmp_path)
    another = execute_recorded_task("data.inspect", payload, project_root=tmp_path,
                                    experiment_id=first["experiment_id"])
    assert first["experiment_id"] == another["experiment_id"]
    assert first["run_id"] != another["run_id"]
    store = ExperimentStore(tmp_path)
    assert len(store.list_runs(first["experiment_id"])) == 2
    assert store.get_experiment(first["experiment_id"])["revision"] == 1
    with pytest.raises(ExperimentStoreError, match="does not match"):
        execute_recorded_task("data.inspect", {"paths": [str(source)], "preview_rows": 9},
                              project_root=tmp_path, experiment_id=first["experiment_id"])
    assert len(store.list_runs(first["experiment_id"])) == 2
    with pytest.raises(ExperimentStoreError, match="requires an experiment_id"):
        execute_recorded_task("data.inspect", payload, project_root=tmp_path, revision=1)


def test_failed_legacy_execution_is_recorded_but_original_api_unchanged(tmp_path: Path):
    payload = {"X": [1, 2, 3]}
    with pytest.raises(PayloadError, match="two-dimensional"):
        execute_task("clustering", payload)
    with pytest.raises(RecordedTaskError) as error:
        execute_recorded_task("clustering", payload, project_root=tmp_path)
    assert isinstance(error.value.__cause__, PayloadError)
    store = ExperimentStore(tmp_path)
    assert store.get_run(error.value.run_id)["status"] == "failed"
    artifacts = store.artifacts.list_for_run(error.value.run_id)
    assert [a["kind"] for a in artifacts] == ["payload"]
    assert store.get_run(error.value.run_id)["error"].startswith("PayloadError:")


def test_invalid_snapshot_is_rejected_before_creating_experiment(tmp_path: Path):
    with pytest.raises(ExperimentStoreError, match="strict JSON"):
        execute_recorded_task("clustering", {"X": [[float("nan")]]}, project_root=tmp_path)
    with pytest.raises(ExperimentStoreError, match="strict JSON"):
        execute_recorded_task("clustering", {"X": [[object()]]}, project_root=tmp_path)
    with pytest.raises(ExperimentStoreError, match="payload keys"):
        execute_recorded_task("clustering", {1: "bad"}, project_root=tmp_path)
    assert not (tmp_path / "analysis").exists()


def test_external_data_path_is_not_registered_or_copied(tmp_path: Path):
    project = tmp_path / "project"
    project.mkdir()
    external = tmp_path / "external.csv"
    external.write_text("a\n1\n", encoding="utf-8")
    response = execute_recorded_task("data.inspect", {"paths": [str(external)]}, project_root=project)
    assert {a["kind"] for a in response["artifacts"]} == {"payload", "result"}
    assert external.exists()
    assert not list(project.rglob("external.csv"))


def test_transform_outputs_and_sidecars_are_registered_without_overwriting(tmp_path: Path):
    source = tmp_path / "original.csv"
    source.write_text("name,value\n A ,1\n B ,2\n", encoding="utf-8")
    derived = tmp_path / "output.csv"
    payload = {"path": str(source),
               "recipe": {"operations": [{"type": "clean_strings", "columns": ["name"], "strip": True}]},
               "output": str(derived)}
    recorded = execute_recorded_task("data.transform.apply", payload, project_root=tmp_path)
    store = ExperimentStore(tmp_path)
    assert store.get_run(recorded["run_id"])["status"] == "succeeded"
    kinds = {a["kind"] for a in recorded["artifacts"]}
    assert {"dataset", "payload", "result", "recipe", "manifest", "provenance"} <= kinds
    output = next(a for a in recorded["artifacts"] if a["kind"] == "manifest")
    assert len(output["source_artifact_ids"]) == 2
    assert json.loads(Path(recorded["response"]["result"]["manifest_path"]).read_text())["schema"] == "ml-lab.derived-dataset@1"
    assert source.read_text().startswith("name,value\n A ")
    assert derived.exists()


def test_sources_link_runs_without_cross_project_registration(tmp_path: Path):
    first = tmp_path / "one"
    second = tmp_path / "two"
    first.mkdir()
    second.mkdir()
    file = first / "input.csv"
    file.write_text("x\n1\n")
    earlier = execute_recorded_task("data.inspect", {"paths": [str(file)]}, project_root=first)
    source_id = next(a for a in earlier["artifacts"] if a["kind"] == "dataset")["artifact_id"]
    later = execute_recorded_task("data.inspect", {"paths": [str(file)]},
                                   project_root=first, source_artifact_ids=[source_id])
    new_input = next(a for a in later["artifacts"] if a["kind"] == "payload")
    assert new_input["source_artifact_ids"] == [source_id]
    with pytest.raises(RecordedTaskError) as exc:
        execute_recorded_task("data.inspect", {"paths": [str(file)]},
                              project_root=second, source_artifact_ids=[source_id])
    assert ExperimentStore(second).get_run(exc.value.run_id)["status"] == "failed"


def test_concurrent_runs_do_not_replace_artifacts(tmp_path: Path):
    source = tmp_path / "input.csv"
    source.write_text("a\n1\n2\n", encoding="utf-8")
    payload = {"paths": [str(source)]}
    first = execute_recorded_task("data.inspect", payload, project_root=tmp_path)
    def one(_: int) -> dict:
        return execute_recorded_task("data.inspect", payload, project_root=tmp_path,
                                     experiment_id=first["experiment_id"])
    with ThreadPoolExecutor(max_workers=4) as workers:
        remaining = list(workers.map(one, range(5)))
    runs = [first, *remaining]
    assert len({r["run_id"] for r in runs}) == len(runs)
    assert len({a["path"] for r in runs for a in r["artifacts"] if a["kind"] == "result"}) == len(runs)
    assert len(ExperimentStore(tmp_path).list_runs(first["experiment_id"])) == 6
