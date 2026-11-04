"""S2.1: Data Lab -> project-local experiment dataset handoff."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ml_lab import data
from ml_lab.experiments import (
    DatasetSelectionError,
    ExperimentStore,
    prepare_model_handoff,
    preview_experiment_dataset,
    select_experiment_dataset,
    verify_selection,
)


def _csv(path: Path, rows: list[list[str]]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as stream:
        csv.writer(stream).writerows(rows)
    return path


def _project(tmp_path: Path, name: str = "project") -> tuple[Path, ExperimentStore]:
    project = tmp_path / name
    project.mkdir()
    return project, ExperimentStore(project)


def test_select_and_prepare_supervised_payload_without_manual_matrix_copy(tmp_path: Path):
    project, store = _project(tmp_path)
    file = _csv(project / "source.csv", [["label", "a", "b"], ["cat", "1", "2"], ["dog", "3", "4"]])
    experiment = select_experiment_dataset(store, file, target_columns=["label"])
    selected = experiment["dataset_ref"]
    assert selected["path"] == "source.csv"
    assert selected["dataset_id"].startswith("dataset:sha256:")
    assert selected["feature_columns"] == ["a", "b"]
    assert selected["target_columns"] == ["label"]
    assert selected["provenance"] is None
    handoff = prepare_model_handoff(store, experiment["experiment_id"], "classification", settings={"estimators": ["ridge"]})
    assert handoff["revision"] == 1
    assert handoff["payload"] == {
        "estimators": ["ridge"], "X": [[1.0, 2.0], [3.0, 4.0]],
        "feature_names": ["a", "b"], "y": ["cat", "dog"], "label_name": "label",
    }
    page = preview_experiment_dataset(store, experiment["experiment_id"], page_size=1)
    assert page["table"]["rows"] == [["cat", "1", "2"]]
    assert page["table"]["matched_row_count"] == 2
    assert file.read_text().startswith("label,a,b")


def test_idempotent_selection_and_revision_reset_preserves_history(tmp_path: Path):
    project, store = _project(tmp_path)
    source = _csv(project / "one.csv", [["a", "b"], ["1", "2"], ["3", "4"]])
    current = select_experiment_dataset(store, source)
    again = select_experiment_dataset(store, source, experiment_id=current["experiment_id"])
    assert again["revision"] == 1
    store.revise_experiment(current["experiment_id"], dataset_ref=current["dataset_ref"],
                            model_spec={"method": "pca"}, evaluation_plan={"metric": "variance"})
    updated = select_experiment_dataset(store, source, feature_columns=["b"], experiment_id=current["experiment_id"])
    assert updated["revision"] == 3
    assert updated["model_spec"] == {}
    assert updated["evaluation_plan"] == {}
    assert store.get_experiment(current["experiment_id"], revision=2)["model_spec"] == {"method": "pca"}
    assert prepare_model_handoff(store, current["experiment_id"], "representation", revision=1)["payload"]["feature_names"] == ["a", "b"]
    assert prepare_model_handoff(store, current["experiment_id"], "representation")["payload"]["feature_names"] == ["b"]


def test_edited_source_is_stale_not_auto_refreshed(tmp_path: Path):
    project, store = _project(tmp_path)
    source = _csv(project / "x.csv", [["x"], ["1"], ["2"]])
    experiment = select_experiment_dataset(store, source)
    _csv(source, [["x"], ["5"], ["6"]])
    with pytest.raises(DatasetSelectionError, match="changed"):
        prepare_model_handoff(store, experiment["experiment_id"], "rbm")
    with pytest.raises(DatasetSelectionError, match="changed"):
        preview_experiment_dataset(store, experiment["experiment_id"])
    assert store.get_experiment(experiment["experiment_id"])["dataset_ref"]["sha256"] != data.inspect_file(source).sha256
    updated = select_experiment_dataset(store, source, experiment_id=experiment["experiment_id"])
    assert updated["revision"] == 2


def test_derived_dataset_preserves_data_lab_lineage(tmp_path: Path):
    project, store = _project(tmp_path)
    original = _csv(project / "input.csv", [["name", "amount"], [" A ", "1"], [" B ", "2"]])
    derived = project / "processed.csv"
    manifest = data.apply_transformation(original, {"operations": [
        {"type": "clean_strings", "columns": ["name"], "strip": True},
    ]}, output=derived)
    selected = select_experiment_dataset(store, manifest["derived"]["path"], feature_columns=["amount"])
    provenance = selected["dataset_ref"]["provenance"]
    assert provenance["event_id"] == manifest["provenance"]["event_id"]
    assert provenance["path"] == "processed.csv.provenance.json"
    assert verify_selection(store, selected["dataset_ref"]) == selected["dataset_ref"]
    (project / provenance["path"]).write_text("{}", encoding="utf-8")
    with pytest.raises(DatasetSelectionError, match="provenance"):
        prepare_model_handoff(store, selected["experiment_id"], "representation")


def test_external_and_symlink_datasets_rejected(tmp_path: Path):
    project, store = _project(tmp_path)
    external = _csv(tmp_path / "external.csv", [["x"], ["1"]])
    with pytest.raises(DatasetSelectionError, match="project-local"):
        select_experiment_dataset(store, external)
    with pytest.raises(DatasetSelectionError, match="project-local"):
        select_experiment_dataset(store, "../external.csv")
    (project / "link.csv").symlink_to(external)
    with pytest.raises(DatasetSelectionError, match="project-local"):
        select_experiment_dataset(store, "link.csv")


def test_unsafe_columns_and_unusable_data_raise_early(tmp_path: Path):
    project, store = _project(tmp_path)
    source = _csv(project / "x.csv", [["a", "label"], ["1", "ok"], ["2", "bad"]])
    with pytest.raises(DatasetSelectionError, match="Unknown"):
        select_experiment_dataset(store, source, feature_columns=["missing"])
    with pytest.raises(DatasetSelectionError, match="reuse"):
        select_experiment_dataset(store, source, feature_columns=["a", "label"], target_columns=["label"])
    with pytest.raises(DatasetSelectionError, match="duplicates"):
        select_experiment_dataset(store, source, feature_columns=["a", "a"])
    with pytest.raises(DatasetSelectionError, match="list"):
        select_experiment_dataset(store, source, feature_columns="a")
    experiment = select_experiment_dataset(store, source, feature_columns=["a"], target_columns=["label"])
    with pytest.raises(DatasetSelectionError, match="does not accept selected targets"):
        prepare_model_handoff(store, experiment["experiment_id"], "representation")
    with pytest.raises(DatasetSelectionError, match="must not replace"):
        prepare_model_handoff(store, experiment["experiment_id"], "classification", settings={"X": [[9]]})
    with pytest.raises(DatasetSelectionError, match="numeric"):
        prepare_model_handoff(store, experiment["experiment_id"], "regression")


def test_numeric_validation_and_clustering_reference_labels(tmp_path: Path):
    project, store = _project(tmp_path)
    source = _csv(project / "x.tsv", [["a", "b", "class"], ["1", "", "ok"], ["2", "3", "bad"]])
    experiment = select_experiment_dataset(store, source, feature_columns=["a", "b"], target_columns=["class"])
    with pytest.raises(DatasetSelectionError, match="missing or nonfinite"):
        prepare_model_handoff(store, experiment["experiment_id"], "clustering")
    _csv(source, [["a", "b", "class"], ["1", "2", "ok"], ["2", "3", "bad"]])
    newer = select_experiment_dataset(store, source, feature_columns=["a", "b"], target_columns=["class"], experiment_id=experiment["experiment_id"])
    handoff = prepare_model_handoff(store, newer["experiment_id"], "clustering")
    assert handoff["payload"]["y_true"] == ["ok", "bad"]
    assert handoff["payload"]["X"] == [[1.0, 2.0], [2.0, 3.0]]


def test_no_implicit_partial_ingest_or_xml_and_project_isolation(tmp_path: Path):
    first, first_store = _project(tmp_path, "first")
    second, second_store = _project(tmp_path, "second")
    csv_file = _csv(first / "raw.csv", [["a", "b"], ["1", "2"], ["3"]])
    with pytest.raises(DatasetSelectionError, match="fully parsed"):
        select_experiment_dataset(first_store, csv_file)
    xml_file = first / "x.xml"
    xml_file.write_text("<root><item>1</item></root>", encoding="utf-8")
    with pytest.raises(DatasetSelectionError, match="fully parsed"):
        select_experiment_dataset(first_store, xml_file)
    valid = _csv(first / "valid.csv", [["a"], ["1"]])
    selected = select_experiment_dataset(first_store, valid)
    with pytest.raises(DatasetSelectionError):
        verify_selection(second_store, selected["dataset_ref"])
    assert not any(second.rglob("*.csv"))


def test_response_suitable_for_existing_dispatcher(tmp_path: Path):
    from ml_lab.application import execute_task
    project, store = _project(tmp_path)
    source = _csv(project / "x.csv", [["a", "b"], ["0", "1"], ["1", "2"], ["2", "3"], ["3", "4"]])
    experiment = select_experiment_dataset(store, source)
    handoff = prepare_model_handoff(store, experiment["experiment_id"], "representation", settings={"method": "pca", "config": {"n_components": 1}})
    result = execute_task(handoff["task"], handoff["payload"])
    assert result["task"] == "representation"
    assert len(result["result"]["latent"]) == 4
