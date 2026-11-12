"""ML-UW-01: a project-local active dataset pointer across workbench sections."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from ml_lab.experiments import (
    ExperimentStore, WorkbenchConflict, WorkbenchInstanceNotFound,
    activate_workbench_experiment, clear_workbench_dataset,
    create_workbench_instance, get_workbench_instance, select_experiment_dataset,
    select_workbench_dataset,
)


def _csv(project: Path, name: str, values: str = "1\n2\n") -> Path:
    source = project / name
    source.write_text("feature\n" + values, encoding="utf-8")
    return source


def test_active_dataset_survives_restart_and_archives_old_revisions(tmp_path):
    store = ExperimentStore(tmp_path)
    first = _csv(tmp_path, "first.csv")
    second = _csv(tmp_path, "second.csv", "3\n4\n")
    context = create_workbench_instance(store)
    assert context["schema"] == "ml-lab.workbench-instance@1"
    assert context["status"] == "empty"
    assert context["generation"] == 0
    active = select_workbench_dataset(store, context["instance_id"], first, expected_generation=0)
    assert active["status"] == "ready"
    assert active["active_dataset"]["path"] == "first.csv"
    old_id = active["experiment_id"]
    archived_run = store.create_run(old_id)
    switched = select_workbench_dataset(store, context["instance_id"], second, expected_generation=1)
    assert switched["generation"] == 2
    assert switched["active_dataset"]["path"] == "second.csv"
    assert ExperimentStore(tmp_path).get_experiment(old_id)["dataset_ref"]["path"] == "first.csv"
    reopened = get_workbench_instance(ExperimentStore(tmp_path), context["instance_id"])
    assert reopened["experiment_id"] == switched["experiment_id"]
    cleared = clear_workbench_dataset(store, context["instance_id"], expected_generation=2)
    assert cleared["status"] == "empty"
    assert cleared["generation"] == 3
    assert len(store.list_experiments()) == 2
    assert store.get_run(archived_run["run_id"])["experiment_id"] == old_id


def test_pin_revision_switching_does_not_revise_experiment(tmp_path):
    store = ExperimentStore(tmp_path)
    first = _csv(tmp_path, "source.csv")
    context = create_workbench_instance(store)
    experiment = select_experiment_dataset(store, first)
    chosen = activate_workbench_experiment(
        store, context["instance_id"], experiment["experiment_id"], expected_generation=0
    )
    assert chosen["experiment_revision"] == 1
    revised = store.revise_experiment(
        experiment["experiment_id"], dataset_ref=experiment["dataset_ref"],
        model_spec={"method": "pca"},
    )
    # The active instance keeps revision 1 until explicitly switched.
    assert get_workbench_instance(store, context["instance_id"])["experiment_revision"] == 1
    later = activate_workbench_experiment(
        store, context["instance_id"], experiment["experiment_id"],
        expected_generation=1, revision=revised["revision"],
    )
    assert later["experiment_revision"] == 2
    assert store.get_experiment(experiment["experiment_id"], revision=1)["model_spec"] == {}


def test_stale_dataset_is_blocked_without_silent_refresh(tmp_path):
    store = ExperimentStore(tmp_path)
    file = _csv(tmp_path, "edited.csv")
    instance = create_workbench_instance(store)
    selected = select_workbench_dataset(store, instance["instance_id"], file, expected_generation=0)
    file.write_text("feature\n9\n10\n", encoding="utf-8")
    stale = get_workbench_instance(store, instance["instance_id"])
    assert stale["status"] == "stale"
    assert stale["active_dataset"] is None
    assert stale["pinned_dataset"]["sha256"] == selected["pinned_dataset"]["sha256"]
    assert "changed" in stale["validation_error"]
    with pytest.raises(ValueError, match="changed"):
        activate_workbench_experiment(store, instance["instance_id"],
                                      selected["experiment_id"], expected_generation=1)
    assert get_workbench_instance(store, instance["instance_id"])["generation"] == 1
    restored = select_workbench_dataset(store, instance["instance_id"], file, expected_generation=1)
    assert restored["status"] == "ready"
    assert restored["active_dataset"]["sha256"] != selected["active_dataset"]["sha256"]


def test_conflicts_and_cross_project_isolation(tmp_path):
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    first_store = ExperimentStore(alpha)
    second_store = ExperimentStore(beta)
    source = _csv(alpha, "a.csv")
    instance = create_workbench_instance(first_store)
    with pytest.raises(WorkbenchInstanceNotFound):
        get_workbench_instance(second_store, instance["instance_id"])
    with pytest.raises(WorkbenchInstanceNotFound):
        get_workbench_instance(first_store, "../first")
    first = select_workbench_dataset(first_store, instance["instance_id"], source, expected_generation=0)
    with pytest.raises(WorkbenchConflict, match="reload"):
        clear_workbench_dataset(first_store, instance["instance_id"], expected_generation=0)
    with pytest.raises(WorkbenchConflict, match="reload"):
        select_workbench_dataset(first_store, instance["instance_id"], source, expected_generation=0)
    assert len(first_store.list_experiments()) == 1
    with pytest.raises(ValueError, match="project-local"):
        select_workbench_dataset(second_store, create_workbench_instance(second_store)["instance_id"],
                                 source, expected_generation=0)
    assert get_workbench_instance(first_store, instance["instance_id"]) == first
    # The backend can already hold independent contexts for future multi-instance UI.
    second_instance = create_workbench_instance(first_store)
    alt = select_workbench_dataset(first_store, second_instance["instance_id"], source,
                                   expected_generation=0)
    assert alt["status"] == "ready"
    assert alt["instance_id"] != first["instance_id"]
    assert first_store.list_experiments()[0]["experiment_id"] in {
        alt["experiment_id"], first["experiment_id"]
    }


def test_upgrade_v2_store_keeps_experiments(tmp_path):
    store = ExperimentStore(tmp_path)
    experiment = store.create_experiment("existing", dataset_ref={"dataset_id": "old"})
    # Simulate an existing v2 database from before ML-UW-01.
    with sqlite3.connect(store.db_path) as db:
        db.execute("DROP TABLE workbench_instances")
        db.execute("PRAGMA user_version = 2")
    reopened = ExperimentStore(tmp_path)
    assert reopened.get_experiment(experiment["experiment_id"]) == experiment
    assert create_workbench_instance(reopened)["status"] == "empty"
    with sqlite3.connect(reopened.db_path) as db:
        assert db.execute("PRAGMA user_version").fetchone()[0] == 3


def test_browser_instance_selection_and_json_endpoints(tmp_path):
    flask = pytest.importorskip("flask")
    from ml_lab.ui import create_ui_blueprint

    project = tmp_path / "browser"
    project.mkdir()
    source = _csv(project, "source.csv")
    app = flask.Flask("ml_uw_01_test")
    app.config["TESTING"] = True
    app.register_blueprint(create_ui_blueprint(name="uw01", experiment_project_root=project),
                           url_prefix="/ml")
    client = app.test_client()
    first = client.get("/ml/api/workbench/context")
    assert first.status_code == 200
    empty = first.get_json()
    assert empty["status"] == "empty"
    assert "HttpOnly" in first.headers["Set-Cookie"]
    assert "SameSite=Lax" in first.headers["Set-Cookie"]
    assert client.get("/ml/api/workbench/context").get_json()["instance_id"] == empty["instance_id"]
    experiment = select_experiment_dataset(ExperimentStore(project), source)
    wrong_generation = client.post("/ml/api/workbench/context/activate", json={
        "experiment_id": experiment["experiment_id"], "expected_generation": 9,
    })
    assert wrong_generation.status_code == 409
    activated = client.post("/ml/api/workbench/context/activate", json={
        "experiment_id": experiment["experiment_id"], "expected_generation": 0,
    })
    assert activated.status_code == 200
    assert activated.get_json()["active_dataset"]["path"] == "source.csv"
    assert client.get("/ml/data").status_code == 200
    assert client.get("/ml/api/workbench/context").get_json()["generation"] == 1
    cleared = client.post("/ml/api/workbench/context/clear", json={"expected_generation": 1})
    assert cleared.status_code == 200
    assert cleared.get_json()["status"] == "empty"
    assert ExperimentStore(project).get_experiment(experiment["experiment_id"])["revision"] == 1
