"""S5.2 plots are artifact-verified and remain independent of Flask."""
from __future__ import annotations

import csv
from io import StringIO
import json
import re

import pytest

from ml_lab.experiments import (
    ExperimentStore, ExperimentStoreError, build_visualization_report,
    chart_csv, chart_svg, configure_experiment_model, execute_configured_run,
    optimize_configured_experiment, queue_configured_run, select_experiment_dataset,
)
from ml_lab.experiments.visualizations import MAX_POINTS, _history_charts


def setup_experiment(tmp_path):
    project = tmp_path / "project"
    project.mkdir()
    source = project / "features.csv"
    source.write_text("a,b,c\n1,2,3\n2,5,6\n3,6,2\n4,9,3\n5,10,6\n6,11,7\n", encoding="utf-8")
    store = ExperimentStore(project)
    exp = select_experiment_dataset(store, source, feature_columns=["a", "b", "c"])
    exp = configure_experiment_model(
        store, exp["experiment_id"], family="representation", method="pca",
        settings={"config": {"n_components": 1}}, expected_revision=exp["revision"],
    )
    return store, exp


def train(store, exp):
    queued = queue_configured_run(store, exp["experiment_id"])
    execute_configured_run(store, queued["run_id"])
    return queued["run_id"]


def fabricated_history_run(store, exp, history, *, metrics=None):
    run = store.create_run(exp["experiment_id"], revision=exp["revision"],
                           environment={"executor": "ml_lab.experiments.training@1"})
    store.update_run_status(run["run_id"], "running")
    dest = store.project_root / f"history-{run['run_id']}.json"
    dest.write_text(json.dumps({"result": {"method": "mlp_autoencoder", "metrics": metrics or {},
                                                "history": history}}), encoding="utf-8")
    artifact = store.artifacts.register(run["run_id"], dest, role="output", kind="result")
    store.update_run_status(run["run_id"], "succeeded")
    return run["run_id"], artifact, dest


def test_comparable_training_runs_have_metrics_bars(tmp_path):
    store, exp = setup_experiment(tmp_path)
    ids = [train(store, exp), train(store, exp)]
    report = build_visualization_report(store, exp["experiment_id"], run_ids=ids)
    bars = [chart for chart in report["charts"] if chart["kind"] == "bar"]
    assert report["comparable"] and bars
    assert any(chart["title"].endswith(" · rmse") for chart in bars)
    assert all(chart["run_ids"] == ids for chart in bars)
    assert all(len(chart["source_artifacts"]) == 2 for chart in bars)
    assert all(len(chart["points"]) == 2 for chart in bars)
    svg = chart_svg(bars[0])
    assert svg.startswith('<svg xmlns="http://www.w3.org/2000/svg"')
    assert '<rect ' in svg and svg.endswith('</svg>')
    assert list(csv.DictReader(StringIO(chart_csv(bars[0]))))[0]["y"]


def test_mixed_training_and_optimization_only_plot_individual_history(tmp_path):
    store, exp = setup_experiment(tmp_path)
    first = train(store, exp)
    plan = {"parameter": "config.n_components", "lower": 1, "upper": 2, "metric": "rmse",
            "goal": "minimize", "population_size": 2, "iterations": 0, "seed": 9}
    optimized = optimize_configured_experiment(store, exp["experiment_id"], plan,
                                               expected_revision=exp["revision"])
    report = build_visualization_report(store, exp["experiment_id"],
                                        run_ids=[first, optimized["run_id"]])
    assert not report["comparable"]
    assert all(chart["kind"] != "bar" for chart in report["charts"])
    trace = next(chart for chart in report["charts"] if chart["id"].startswith("optimization-"))
    assert trace["y_label"] == "rmse"
    assert len(trace["points"]) == 2
    assert all(isinstance(p["y"], float) for p in trace["points"])
    assert len(trace["source_artifacts"]) == 1
    assert 'fitness' not in trace["title"].lower()
    assert 'polyline' in chart_svg(trace)


def test_model_history_epoch_curve_is_not_inferred_from_lifecycle(tmp_path):
    store, exp = setup_experiment(tmp_path)
    run_id, artifact, _ = fabricated_history_run(store, exp, [
        {"epoch": 1, "loss": 2., "unavailable": None},
        {"epoch": 2, "loss": 1.5},
        {"epoch": 3, "loss": 1.25},
    ])
    report = build_visualization_report(store, exp["experiment_id"], run_ids=[run_id])
    assert len(report["charts"]) == 1
    chart = report["charts"][0]
    assert chart["points"] == [
        {"x": 1, "y": 2., "label": "1"}, {"x": 2, "y": 1.5, "label": "2"},
        {"x": 3, "y": 1.25, "label": "3"},
    ]
    assert chart["source_artifacts"] == [{"artifact_id": artifact["artifact_id"], "sha256": artifact["sha256"]}]


def test_history_sampling_bounded_and_preserves_last_epoch():
    history = [{"epoch": i, "loss": 1 / (i + 1)} for i in range(2500)]
    charts = _history_charts({"result": {"history": history}}, run_id="a" * 32,
                             artifact={"artifact_id": "s", "sha256": "v"}, allowance=4)
    assert len(charts) == 1
    assert len(charts[0]["points"]) == MAX_POINTS
    assert charts[0]["points"][0]["x"] == 0
    assert charts[0]["points"][-1]["x"] == 2499
    assert "sampled" in charts[0]["note"]


def test_tampering_suppresses_visualizations(tmp_path):
    store, exp = setup_experiment(tmp_path)
    run_id, artifact, path = fabricated_history_run(store, exp,
                                                    [{"epoch": 1, "loss": 2}, {"epoch": 2, "loss": 1}])
    assert build_visualization_report(store, exp["experiment_id"], run_ids=[run_id])["charts"]
    path.write_text(path.read_text().replace('"loss": 1', '"loss": 999'))
    report = build_visualization_report(store, exp["experiment_id"], run_ids=[run_id])
    assert not report["charts"]
    assert any("integrity" in n for n in report["notes"])


def test_missing_optimizer_history_not_substituted_with_fitness(tmp_path):
    store, exp = setup_experiment(tmp_path)
    plan = {"parameter": "config.n_components", "lower": 1, "upper": 2, "metric": "rmse",
            "goal": "minimize", "population_size": 2, "iterations": 0, "seed": 9}
    result = optimize_configured_experiment(store, exp["experiment_id"], plan,
                                            expected_revision=exp["revision"])
    report = build_visualization_report(store, exp["experiment_id"], run_ids=[result["run_id"]])
    assert len(report["charts"]) == 1
    history = next(a for a in store.artifacts.list_for_run(result["run_id"])
                   if a["name"] == "optimization_history.csv")
    path = store.project_root / history["path"]
    path.write_text(path.read_text() + '\nINVALID\n')
    report = build_visualization_report(store, exp["experiment_id"], run_ids=[result["run_id"]])
    assert not report["charts"]
    assert report["notes"]


def test_no_cross_experiment_or_cross_project_selection(tmp_path):
    store, exp = setup_experiment(tmp_path)
    first = train(store, exp)
    other = store.create_experiment("other")
    other_run = store.create_run(other["experiment_id"], environment={"executor": "ml_lab.experiments.training@1"})
    with pytest.raises(ExperimentStoreError):
        build_visualization_report(store, exp["experiment_id"], run_ids=[first, other_run["run_id"]])
    other_project = tmp_path / "second"
    other_project.mkdir()
    with pytest.raises(ExperimentStoreError):
        build_visualization_report(ExperimentStore(other_project), exp["experiment_id"], run_ids=[first])


def test_svg_escapes_untrusted_text_and_csv_escapes_formulas():
    malicious = {"title": '<svg onload="alert(1)">', "kind": "bar", "x_label": '"<script>',
                 "y_label": "m", "points": [{"x": 0, "label": '=SUM(A1)', "y": -3.25}]}
    svg = chart_svg(malicious)
    assert 'onload="alert(1)"' not in svg
    assert '&lt;script&gt;' in svg
    assert not re.search(r'<script|<svg onload', svg)
    parsed = list(csv.DictReader(StringIO(chart_csv(malicious))))
    assert parsed[0]["label"] == "'=SUM(A1)"
    assert parsed[0]["y"] == "-3.25"
