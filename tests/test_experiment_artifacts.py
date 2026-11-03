"""S1.2 artifact registry contracts and migration safety."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sqlite3

import pytest

from ml_lab.experiments import (
    ArtifactError,
    ArtifactNotFound,
    ExperimentNotFound,
    ExperimentStore,
    ExperimentStoreError,
    SchemaVersionError,
)


def _create_run(store: ExperimentStore) -> tuple[str, str]:
    exp = store.create_experiment("provenance", dataset_ref={"source": "local"})
    run = store.create_run(exp["experiment_id"], seed=4)
    return exp["experiment_id"], run["run_id"]


def test_register_reopen_and_verify_artifacts_and_sources(tmp_path: Path):
    store = ExperimentStore(tmp_path)
    experiment_id, first_run = _create_run(store)
    source_path = tmp_path / "datasets" / "source.csv"
    source_path.parent.mkdir()
    source_path.write_text("x,y\n1,2\n", encoding="utf-8")
    source = store.artifacts.register(
        first_run, source_path, role="input", kind="dataset", media_type="text/csv",
        metadata={"original": True},
    )
    assert source["schema"] == "ml-lab.artifact@1"
    assert source["path"] == "datasets/source.csv"
    assert source["experiment_id"] == experiment_id
    assert source["revision"] == 1
    assert source["source_artifact_ids"] == []
    assert source["sha256"] and len(source["sha256"]) == 64
    assert store.artifacts.verify(source["artifact_id"])["status"] == "valid"

    second_run = store.create_run(experiment_id)["run_id"]
    product_path = tmp_path / "analysis" / "derived.csv"
    product_path.parent.mkdir(exist_ok=True)
    product_path.write_text("x,y\n3,4\n", encoding="utf-8")
    output = store.artifacts.register(
        second_run, "analysis/derived.csv", kind="dataset", name="cleaned",
        source_artifact_ids=[source["artifact_id"]],
        metadata={"transformation": "replace-values"},
    )
    reopened = ExperimentStore(tmp_path)
    assert reopened.artifacts.get(output["artifact_id"]) == output
    assert reopened.artifacts.list_for_run(second_run) == [output]
    assert len(reopened.artifacts.list_for_experiment(experiment_id)) == 2
    assert output["source_artifact_ids"] == [source["artifact_id"]]
    assert reopened.artifacts.verify(output["artifact_id"])["status"] == "valid"

    product_path.write_text("x,y\n5,6\n", encoding="utf-8")
    assert reopened.artifacts.verify(output["artifact_id"])["status"] == "changed"
    assert reopened.artifacts.get(output["artifact_id"]) == output  # immutable fingerprint
    product_path.unlink()
    assert reopened.artifacts.verify(output["artifact_id"])["status"] == "missing"


def test_validation_and_atomicity(tmp_path: Path):
    store = ExperimentStore(tmp_path)
    experiment_id, run_id = _create_run(store)
    file_path = tmp_path / "log.txt"
    file_path.write_text("log", encoding="utf-8")
    with pytest.raises(ArtifactError):
        store.artifacts.register(run_id, file_path, role="other")
    with pytest.raises(ArtifactError):
        store.artifacts.register(run_id, file_path, kind="Uppercase")
    with pytest.raises(ExperimentStoreError):
        store.artifacts.register(run_id, file_path, metadata={"bad": object()})
    with pytest.raises(ArtifactError):
        store.artifacts.register(run_id, file_path, source_artifact_ids="invalid")
    with pytest.raises(ArtifactNotFound):
        store.artifacts.register(run_id, file_path, source_artifact_ids=["missing"])
    with pytest.raises(ExperimentNotFound):
        store.artifacts.register("missing", file_path)
    assert store.artifacts.list_for_run(run_id) == []
    with pytest.raises(ExperimentNotFound):
        store.artifacts.list_for_run("missing")
    with pytest.raises(ExperimentNotFound):
        store.artifacts.list_for_experiment("missing")
    with pytest.raises(ArtifactNotFound):
        store.artifacts.get("missing")
    with pytest.raises(ExperimentStoreError):
        store.artifacts.list_for_experiment(experiment_id, limit=501)
    with pytest.raises(ExperimentStoreError):
        store.artifacts.list_for_run(run_id, offset=-1)

    first = store.artifacts.register(run_id, file_path)
    with pytest.raises(ArtifactError):
        store.artifacts.register(run_id, file_path, source_artifact_ids=[first["artifact_id"]] * 2)
    assert store.artifacts.list_for_run(run_id) == [first]


def test_project_isolation_and_symlink_rejection(tmp_path: Path):
    pa, pb = tmp_path / "a", tmp_path / "b"
    pa.mkdir()
    pb.mkdir()
    primary, other = ExperimentStore(pa), ExperimentStore(pb)
    _, run_a = _create_run(primary)
    _, run_b = _create_run(other)
    (pa / "one.txt").write_text("a")
    (pb / "two.txt").write_text("b")
    source = primary.artifacts.register(run_a, "one.txt")
    with pytest.raises(ArtifactError):
        primary.artifacts.register(run_a, pb / "two.txt")
    with pytest.raises(ArtifactError):
        primary.artifacts.register(run_a, "../b/two.txt")
    with pytest.raises(ArtifactError):
        primary.artifacts.register(run_a, "missing.txt")
    (pa / "link.txt").symlink_to(pb / "two.txt")
    with pytest.raises(ArtifactError):
        primary.artifacts.register(run_a, "link.txt")
    with pytest.raises(ArtifactNotFound):
        other.artifacts.register(run_b, "two.txt", source_artifact_ids=[source["artifact_id"]])
    with pytest.raises(ArtifactNotFound):
        other.artifacts.get(source["artifact_id"])

    # A registered file changed into an external symlink is never treated as valid.
    (pa / "one.txt").unlink()
    (pa / "one.txt").symlink_to(pb / "two.txt")
    assert primary.artifacts.verify(source["artifact_id"])["status"] == "unsafe"


def test_existing_v1_db_migrates_without_losing_runs(tmp_path: Path):
    store = ExperimentStore(tmp_path)
    exp, run = _create_run(store)
    store.update_run_status(run, "succeeded")
    # Simulate the exact tables/version left by S1.1; preserve their data.
    with sqlite3.connect(store.db_path) as db:
        db.execute("DROP TABLE artifact_sources")
        db.execute("DROP TABLE artifacts")
        db.execute("PRAGMA user_version = 1")
    migrated = ExperimentStore(tmp_path)
    assert migrated.get_run(run)["status"] == "succeeded"
    assert migrated.get_experiment(exp)["revision"] == 1
    assert migrated.artifacts.list_for_run(run) == []
    with sqlite3.connect(migrated.db_path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 2
    (tmp_path / "out.txt").write_text("ok")
    assert migrated.artifacts.register(run, "out.txt")["path"] == "out.txt"
    with sqlite3.connect(migrated.db_path) as db:
        db.execute("PRAGMA user_version = 99")
    with pytest.raises(SchemaVersionError):
        ExperimentStore(tmp_path)


def test_parallel_writers_and_pagination(tmp_path: Path):
    store = ExperimentStore(tmp_path)
    exp, run = _create_run(store)
    path = tmp_path / "results.json"
    path.write_text("{}")

    def register(index: int) -> str:
        return ExperimentStore(tmp_path).artifacts.register(run, path, name=f"result-{index}")["artifact_id"]

    with ThreadPoolExecutor(max_workers=5) as pool:
        ids = list(pool.map(register, range(12)))
    assert len(set(ids)) == 12
    first_page = store.artifacts.list_for_experiment(exp, limit=5)
    next_page = store.artifacts.list_for_experiment(exp, limit=5, offset=5)
    assert len(first_page) == len(next_page) == 5
    assert set(a["artifact_id"] for a in first_page).isdisjoint(
        a["artifact_id"] for a in next_page
    )
