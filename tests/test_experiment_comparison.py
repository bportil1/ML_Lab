"""S5.1: verified read-only evaluation and cross-run comparison."""
from __future__ import annotations

import csv
from io import StringIO
import json

import pytest

from ml_lab.experiments import (
    ExperimentStore, ExperimentStoreError, compare_experiment_runs,
    comparison_csv, configure_experiment_model, execute_configured_run,
    optimize_configured_experiment, queue_configured_run,
    select_experiment_dataset,
)


def prepared(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    source = project / "features.csv"
    source.write_text("a,b,c\n1,2,3\n2,5,6\n3,6,2\n4,9,3\n5,10,6\n6,11,7\n", encoding="utf-8")
    store = ExperimentStore(project)
    item = select_experiment_dataset(store, source, feature_columns=["a", "b", "c"])
    item = configure_experiment_model(store, item["experiment_id"], family="representation",
                                      method="pca", settings={"config": {"n_components": 1}},
                                      expected_revision=item["revision"])
    return store, item, source


def train(store, item):
    queued = queue_configured_run(store, item["experiment_id"], revision=item["revision"])
    execute_configured_run(store, queued["run_id"])
    return queued["run_id"]


def test_real_pca_and_optimizer_comparison(tmp_path):
    store, item, _ = prepared(tmp_path)
    training_id = train(store, item)
    plan = {"parameter": "config.n_components", "lower": 1, "upper": 2,
            "metric": "rmse", "goal": "minimize", "population_size": 2,
            "iterations": 0, "seed": 9}
    optimized = optimize_configured_experiment(store, item["experiment_id"], plan,
                                               expected_revision=item["revision"])
    report = compare_experiment_runs(store, item["experiment_id"],
                                     run_ids=[training_id, optimized["run_id"]])
    assert len(report["runs"]) == 2
    model, trial = report["runs"]
    assert model["type"] == "model" and model["status"] == "succeeded"
    assert any(k.endswith(" · rmse") for k in model["metrics"])
    assert trial["type"] == "optimization"
    assert trial["metrics"]["best observed · rmse"] == pytest.approx(
        optimized["best_fitness"])
    assert trial["optimization"]["evaluations"] == 2
    assert not report["comparable"]  # Optimizer objectives are not direct model scores.
    assert "must not be ranked" in report["warning"]
    parsed = list(csv.DictReader(StringIO(comparison_csv(report))))
    assert len(parsed) == 2 and parsed[0]["run_id"] == training_id
    assert parsed[1]["best observed · rmse"]
    assert store.get_experiment(item["experiment_id"])["revision"] == item["revision"]


def test_no_metric_invention_for_queued_failed_and_cancelled(tmp_path):
    store, item, _ = prepared(tmp_path)
    queued = queue_configured_run(store, item["experiment_id"])
    failed = queue_configured_run(store, item["experiment_id"])
    store.update_run_status(failed["run_id"], "running")
    store.update_run_status(failed["run_id"], "failed", error="deliberate failure")
    cancelled = queue_configured_run(store, item["experiment_id"])
    store.update_run_status(cancelled["run_id"], "cancelled")
    report = compare_experiment_runs(store, item["experiment_id"],
                                     run_ids=[queued["run_id"], failed["run_id"], cancelled["run_id"]])
    assert all(not row["metrics"] and row["result_note"] == "Not complete" for row in report["runs"])
    assert report["runs"][1]["error"] == "deliberate failure"
    assert not report["metric_columns"]


def test_modified_result_is_not_read(tmp_path):
    store, item, _ = prepared(tmp_path)
    run_id = train(store, item)
    result = next(a for a in store.artifacts.list_for_run(run_id) if a["kind"] == "result")
    path = store.project_root / result["path"]
    contents = json.loads(path.read_text())
    contents["result"]["metrics"]["rmse"] = -99999
    path.write_text(json.dumps(contents))
    row = compare_experiment_runs(store, item["experiment_id"], run_ids=[run_id])["runs"][0]
    assert row["metrics"] == {}
    assert row["result_note"] == "Result integrity check failed"


def test_snapshots_differing_dataset_are_not_claimed_comparable(tmp_path):
    store, item, source = prepared(tmp_path)
    first_id = train(store, item)
    # A distinct, immutable dataset selection is still a valid revision.
    alternative = store.project_root / "other.csv"
    alternative.write_text(source.read_text().replace("1,2,3", "2,2,3"), encoding="utf-8")
    later = select_experiment_dataset(store, alternative, experiment_id=item["experiment_id"],
                                      feature_columns=["a", "b", "c"], expected_revision=item["revision"])
    later = configure_experiment_model(store, later["experiment_id"], family="representation",
                                       method="pca", settings={"config": {"n_components": 1}},
                                       expected_revision=later["revision"])
    second_id = train(store, later)
    report = compare_experiment_runs(store, item["experiment_id"], run_ids=[first_id, second_id])
    assert not report["comparable"]
    assert report["runs"][0]["dataset_sha256"] != report["runs"][1]["dataset_sha256"]
    assert [r["revision"] for r in report["runs"]] == [item["revision"], later["revision"]]


def test_reject_other_experiments_projects_and_bad_selection(tmp_path):
    store, item, _ = prepared(tmp_path)
    run_id = train(store, item)
    other = store.create_experiment("unrelated")
    unrelated_id = store.create_run(other["experiment_id"])["run_id"]
    with pytest.raises(ExperimentStoreError):
        compare_experiment_runs(store, item["experiment_id"], run_ids=[run_id, unrelated_id])
    with pytest.raises(ExperimentStoreError):
        compare_experiment_runs(store, item["experiment_id"], run_ids=[run_id, run_id])
    with pytest.raises(ExperimentStoreError):
        compare_experiment_runs(store, item["experiment_id"], run_ids=[run_id] * 26)
    with pytest.raises(ExperimentStoreError):
        compare_experiment_runs(store, item["experiment_id"], limit=26)
    outside = tmp_path / "outside"
    outside.mkdir()
    foreign = ExperimentStore(outside)
    with pytest.raises(ExperimentStoreError):
        compare_experiment_runs(foreign, item["experiment_id"], run_ids=[run_id])


def test_model_metric_extractor_filters_nonfinite_and_bool():
    from ml_lab.experiments.comparison import _model_metrics, _optimization_metrics
    example = {"results": [{"estimator_id": "ridge", "metrics": {
        "r2": 0.5, "bad": float("nan"), "pass": True, "array": [1, 2], "count": 3}}]}
    assert _model_metrics(example) == {"ridge · r2": .5, "ridge · count": 3}
    assert _optimization_metrics({"algorithm": "firefly", "best_objective_terms": {"r2": .7}},
                                 {"metric": "r2", "goal": "maximize"})[0] == {"best observed · r2": .7}
