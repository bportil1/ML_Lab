"""S2.3 — explicit Data Lab/matrix draft to experiment revision handoff."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from ml_lab import data
from ml_lab.experiments import (
    DatasetSelectionError,
    ExperimentStore,
    ExperimentStoreError,
    prepare_model_handoff,
    select_experiment_dataset,
    verify_selection,
)


def _source(project: Path) -> Path:
    path = project / "input.csv"
    path.write_text("a,b,label\n1,2,cat\n3,4,dog\n", encoding="utf-8")
    return path


def _matrix(project: Path, source: Path) -> Path:
    result = data.commit_matrix_draft(
        source, output_root=project / "analysis/data/derived",
        expected_sha256=data.inspect_file(source).sha256,
        edits=[{"row": 1, "column": 0, "value": "11"}],
    )
    return Path(result["path"])


def test_matrix_draft_can_replace_selection_without_retyping_matrix(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    source = _source(project)
    store = ExperimentStore(project)
    old = select_experiment_dataset(store, source, target_columns=["label"])
    # A configured experiment still retains its immutable previous revision.
    configured = store.revise_experiment(
        old["experiment_id"], dataset_ref=old["dataset_ref"],
        model_spec={"model": "old"}, evaluation_plan={"metric": "old"},
    )
    derived = _matrix(project, source)
    updated = select_experiment_dataset(
        store, derived, experiment_id=old["experiment_id"],
        feature_columns=["a", "b"], target_columns=["label"],
        expected_revision=configured["revision"],
    )
    assert updated["revision"] == 3
    assert updated["dataset_ref"]["path"].startswith("analysis/data/derived/")
    assert updated["dataset_ref"]["provenance"]["event_id"] == data.load_provenance_event(derived)["event_id"]
    assert updated["model_spec"] == {}
    assert updated["evaluation_plan"] == {}
    assert configured["model_spec"] == store.get_experiment(old["experiment_id"], revision=2)["model_spec"]
    assert prepare_model_handoff(store, old["experiment_id"], "classification", revision=1)["payload"]["X"] == [[1.0, 2.0], [3.0, 4.0]]
    assert prepare_model_handoff(store, old["experiment_id"], "classification")["payload"]["X"] == [[11.0, 2.0], [3.0, 4.0]]
    assert source.read_text() == "a,b,label\n1,2,cat\n3,4,dog\n"


def test_transformed_dataset_with_changed_columns_requires_explicit_mapping(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    source = _source(project)
    store = ExperimentStore(project)
    current = select_experiment_dataset(store, source, target_columns=["label"])
    output = project / "analysis/data/derived/renamed.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    transformed = data.apply_transformation(source, {"operations": [
        {"type": "rename_columns", "mapping": {"a": "new_a"}},
    ]}, output=output)
    with pytest.raises(DatasetSelectionError, match="Unknown"):
        select_experiment_dataset(store, transformed["derived"]["path"],
                                  experiment_id=current["experiment_id"],
                                  feature_columns=["a", "b"], target_columns=["label"],
                                  expected_revision=1)
    assert store.get_experiment(current["experiment_id"])["revision"] == 1
    selected = select_experiment_dataset(store, output,
                                         experiment_id=current["experiment_id"],
                                         feature_columns=["new_a", "b"], target_columns=["label"],
                                         expected_revision=1)
    assert selected["revision"] == 2
    assert selected["dataset_ref"]["provenance"] is not None
    assert verify_selection(store, selected["dataset_ref"])


def test_stale_revision_does_not_overwrite_new_dataset(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    source = _source(project)
    store = ExperimentStore(project)
    first = select_experiment_dataset(store, source)
    derived = _matrix(project, source)
    second = select_experiment_dataset(store, derived,
                                       experiment_id=first["experiment_id"],
                                       expected_revision=1)
    assert second["revision"] == 2
    with pytest.raises(DatasetSelectionError, match="revision changed"):
        select_experiment_dataset(store, source,
                                  experiment_id=first["experiment_id"],
                                  expected_revision=1)
    with pytest.raises(DatasetSelectionError, match="revision changed"):
        select_experiment_dataset(store, derived,
                                  experiment_id=first["experiment_id"],
                                  expected_revision=1)
    assert store.get_experiment(first["experiment_id"])["dataset_ref"] == second["dataset_ref"]


def test_atomic_revision_compare_and_swap(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    source = _source(project)
    store = ExperimentStore(project)
    current = select_experiment_dataset(store, source)
    def update(feature):
        return store.revise_experiment(current["experiment_id"],
                                       dataset_ref=current["dataset_ref"],
                                       model_spec={"feature": feature}, expected_revision=1)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(update, name) for name in ("a", "b")]
        outcomes = []
        for future in futures:
            try:
                outcomes.append(future.result())
            except ExperimentStoreError as exc:
                assert "revision changed" in str(exc)
    assert len(outcomes) == 1
    assert store.get_experiment(current["experiment_id"])["revision"] == 2


def test_derived_lineage_change_and_external_paths_are_rejected(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    other = tmp_path / "other"
    other.mkdir()
    source = _source(project)
    derived = _matrix(project, source)
    store = ExperimentStore(project)
    selected = select_experiment_dataset(store, derived)
    provenance = data.provenance_sidecar_path(derived)
    provenance.write_text(provenance.read_text().replace('"authoritative": true', '"authoritative": false'), encoding="utf-8")
    with pytest.raises(DatasetSelectionError, match="provenance"):
        verify_selection(store, selected["dataset_ref"])
    with pytest.raises(DatasetSelectionError):
        select_experiment_dataset(ExperimentStore(other), derived)


def test_old_clients_remain_unchanged_and_expected_revision_is_opt_in(tmp_path):
    project = tmp_path / "proj"
    project.mkdir()
    source = _source(project)
    store = ExperimentStore(project)
    first = select_experiment_dataset(store, source)
    assert select_experiment_dataset(store, source,
                                     experiment_id=first["experiment_id"])["revision"] == 1
    with pytest.raises(DatasetSelectionError, match="only valid"):
        select_experiment_dataset(store, source, expected_revision=1)
    with pytest.raises(ExperimentStoreError, match="positive integer"):
        store.revise_experiment(first["experiment_id"], expected_revision=True)
