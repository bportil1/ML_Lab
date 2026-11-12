"""S5.2 read-only, artifact-verified scientific charts and portable exports.

Only persisted measurements are plotted. Never infer epoch curves from the
S4 stage-only progress file, re-evaluate a model, or compare mixed cohorts.
SVG rendering uses only the standard library; no external script/CDN needed.
"""
from __future__ import annotations

import csv
import hashlib
from html import escape
from io import StringIO
import json
import math
from typing import Any, Mapping, Sequence

from .comparison import _number, _result_document, compare_experiment_runs
from .optimization import OPTIMIZER_EXECUTOR
from .store import ExperimentStore, ExperimentStoreError

VISUALIZATION_SCHEMA = "ml-lab.experiment-visualizations@1"
MAX_CHARTS = 24
MAX_POINTS = 500
MAX_HISTORY_ROWS = 10000
MAX_HISTORY_BYTES = 5 * 1024 * 1024


def _chart(identifier: str, title: str, *, kind: str, x_label: str, y_label: str,
           points: list[dict[str, Any]], run_ids: list[str], sources: list[dict[str, str]],
           note: str = "") -> dict[str, Any]:
    return {
        "id": identifier, "title": title, "kind": kind, "x_label": x_label,
        "y_label": y_label, "points": points, "run_ids": run_ids,
        "source_artifacts": sources, "note": note,
    }


def _sample(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deterministic downsampling; always preserve the last observation."""
    if len(rows) <= MAX_POINTS:
        return rows
    return [rows[(i * (len(rows) - 1)) // (MAX_POINTS - 1)] for i in range(MAX_POINTS)]


def _history_charts(document: Any, *, run_id: str, artifact: Mapping[str, Any],
                    allowance: int) -> list[dict[str, Any]]:
    if not isinstance(document, Mapping):
        return []
    records = document.get("results")
    if not isinstance(records, list):
        records = [document.get("result")]
    charts = []
    for idx, result in enumerate(records[:8]):
        if not isinstance(result, Mapping):
            continue
        history = result.get("history")
        if not isinstance(history, list) or not history or len(history) > MAX_HISTORY_ROWS:
            continue
        # Native trainer history uses epoch (some implementations use iteration).
        # Never invent an epoch counter for records that lack an x coordinate.
        axis = "epoch" if all(isinstance(r, dict) and _number(r.get("epoch")) is not None
                              for r in history) else "iteration"
        if not all(isinstance(r, dict) and _number(r.get(axis)) is not None for r in history):
            continue
        name = result.get("estimator_id") or result.get("method") or result.get("scheme") or f"model_{idx + 1}"
        name = str(name)[:80]
        candidates = []
        for key in history[0]:
            if key == axis or not isinstance(key, str) or len(key) > 80:
                continue
            samples = [r for r in history if _number(r.get(key)) is not None]
            if len(samples) < 2:
                continue
            candidates.append((key, samples))
        for key, samples in candidates[:min(4, allowance - len(charts))]:
            points = [{"x": r[axis], "y": r[key], "label": str(r[axis])} for r in samples]
            charts.append(_chart(
                f"history-{run_id}-{idx}-{len(charts)}", f"{name} · {key}",
                kind="line", x_label=axis.title(), y_label=key, points=_sample(points),
                run_ids=[run_id], sources=[{"artifact_id": artifact["artifact_id"], "sha256": artifact["sha256"]}],
                note=("Recorded trainer history; uniformly sampled for display" if len(points) > MAX_POINTS
                      else "Recorded trainer history"),
            ))
        if len(charts) >= allowance:
            break
    return charts


def _verified_optimizer_csv(store: ExperimentStore, run_id: str) -> tuple[list[dict[str, str]], dict[str, str]] | None:
    artifacts = [a for a in store.artifacts.list_for_run(run_id, limit=500)
                 if a["role"] == "output" and a["kind"] == "history"
                 and a["name"] == "optimization_history.csv"]
    if len(artifacts) != 1 or artifacts[0]["size_bytes"] > MAX_HISTORY_BYTES:
        return None
    artifact = artifacts[0]
    try:
        if store.artifacts.verify(artifact["artifact_id"])["status"] != "valid":
            return None
        path, _ = store.artifacts._project_file(store.project_root / artifact["path"])
        if path.stat().st_size > MAX_HISTORY_BYTES:
            return None
        # Read under a hard byte bound, and never trust an unregistered file.
        with path.open("rb") as handle:
            raw = handle.read(MAX_HISTORY_BYTES + 1)
        if len(raw) > MAX_HISTORY_BYTES or hashlib.sha256(raw).hexdigest() != artifact["sha256"]:
            return None
        parser = csv.DictReader(StringIO(raw.decode("utf-8")))
        rows = []
        for row in parser:
            if len(rows) >= MAX_HISTORY_ROWS:
                return None
            rows.append(row)
        return rows, {"artifact_id": artifact["artifact_id"], "sha256": artifact["sha256"]}
    except (OSError, UnicodeError, ValueError, ExperimentStoreError, csv.Error):
        return None


def _optimization_chart(store: ExperimentStore, run_id: str, plan: Mapping[str, Any]):
    metric = plan.get("metric")
    if not isinstance(metric, str) or not metric or len(metric) > 80:
        return None
    result = _verified_optimizer_csv(store, run_id)
    if result is None:
        return None
    rows, artifact = result
    points = []
    for index, row in enumerate(rows, start=1):
        # The fitness value may be goal-adjusted; use measured objective only.
        if row.get("status") != "ok":
            continue
        try:
            observed = float(row.get("objective." + metric, ""))
        except (TypeError, ValueError):
            continue
        if not math.isfinite(observed):
            continue
        points.append({"x": index, "y": observed, "label": str(index)})
    if not points:
        return None
    algorithm = plan.get("algorithm", "firefly")
    title = "Random Search" if algorithm == "random_search" else "Firefly"
    return _chart(
        f"optimization-{run_id}", f"{title} · observed {metric}",
        kind="line", x_label="Candidate evaluation (recorded order)",
        y_label=metric, points=_sample(points), run_ids=[run_id], sources=[artifact],
        note="Observed objective per successful candidate; failed trials omitted. "
             + ("Uniformly sampled for display." if len(points) > MAX_POINTS else "No fitness-to-metric conversion."),
    )


def build_visualization_report(
    store: ExperimentStore, experiment_id: str, *, run_ids: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Describe charts using the same run selection and integrity rules as S5.1."""
    comparison = compare_experiment_runs(store, experiment_id, run_ids=run_ids)
    charts: list[dict[str, Any]] = []
    notes: list[str] = []
    rows = comparison["runs"]
    if comparison["comparable"] and len(rows) >= 2 and all(r["type"] == "model" for r in rows):
        for metric in comparison["metric_columns"][:8]:
            measured = [r for r in rows if metric in r["metrics"]]
            if len(measured) < 2:
                continue
            sources = []
            points = []
            for run in measured:
                artifact = next((a for a in store.artifacts.list_for_run(run["run_id"], limit=500)
                                 if a["role"] == "output" and a["kind"] == "result"), None)
                if artifact is None:
                    continue
                sources.append({"artifact_id": artifact["artifact_id"], "sha256": artifact["sha256"]})
                points.append({"x": len(points), "label": run["run_id"][:10], "y": run["metrics"][metric]})
            if len(points) >= 2:
                charts.append(_chart(
                    f"metric-{len(charts)}", metric, kind="bar", x_label="Run ID (prefix)",
                    y_label=metric, points=points,
                    run_ids=[r["run_id"] for r in measured], sources=sources,
                    note="Comparable experiment revisions only; no metric normalization or ranking.",
                ))
    elif len(rows) > 1 and comparison["warning"]:
        notes.append("Mixed cohorts: cross-run metric plots suppressed. Inspect individual run histories instead.")
    elif len(rows) > 1 and all(r["type"] == "optimization" for r in rows):
        notes.append("Optimizer runs are shown as independent objective traces; search plans and budgets may differ.")
    for row in rows:
        if len(charts) >= MAX_CHARTS:
            break
        if row["status"] != "succeeded":
            continue
        if row["result_note"] not in ("ok", "No supported scalar evaluation metrics"):
            notes.append(f"{row['run_id'][:10]}: result is not integrity-verified ({row['result_note']}).")
            continue
        if row["type"] == "optimization":
            run = store.get_run(row["run_id"])
            chart = _optimization_chart(store, row["run_id"], run["environment"].get("plan", {}))
            if chart is not None:
                charts.append(chart)
            else:
                notes.append(f"{row['run_id'][:10]}: no verified optimizer objective history available.")
        else:
            document, status = _result_document(store, row["run_id"])
            if status != "ok":
                notes.append(f"{row['run_id'][:10]}: recorded result unavailable ({status}).")
                continue
            artifact = next((a for a in store.artifacts.list_for_run(row["run_id"], limit=500)
                             if a["role"] == "output" and a["kind"] == "result"), None)
            if artifact is not None:
                charts.extend(_history_charts(document, run_id=row["run_id"], artifact=artifact,
                                              allowance=min(4, MAX_CHARTS - len(charts))))
    return {
        "schema": VISUALIZATION_SCHEMA, "experiment_id": comparison["experiment_id"],
        "experiment_name": comparison["experiment_name"],
        "selected_runs": [r["run_id"] for r in rows],
        "comparable": comparison["comparable"], "warning": comparison["warning"],
        "charts": charts[:MAX_CHARTS], "notes": notes,
        "policy": "Read-only charts; verified project-local result/history artifacts; no recalculation of scientific metrics",
    }


def chart_csv(chart: Mapping[str, Any]) -> str:
    """Raw plotted points, with spreadsheet-formula-safe textual columns."""
    stream = StringIO(newline="")
    writer = csv.writer(stream)
    writer.writerow(["x", "label", "y"])
    for point in chart["points"]:
        label = str(point["label"])
        if label.lstrip().startswith(("=", "+", "-", "@")):
            label = "'" + label
        writer.writerow((point["x"], label, point["y"]))
    return stream.getvalue()


def chart_svg(chart: Mapping[str, Any]) -> str:
    """Generate standalone neutral SVG with real numeric axes (no JS, no CSS CDN)."""
    points = chart["points"]
    if not points:
        raise ExperimentStoreError("Chart has no recorded points")
    width, height = 900, 340
    left, right, top, bottom = 86, 22, 42, 74
    plot_w, plot_h = width - left - right, height - top - bottom
    values = [float(p["y"]) for p in points]
    if not all(math.isfinite(y) for y in values):
        raise ExperimentStoreError("Non-finite values are not plottable")
    is_bar = chart["kind"] == "bar"
    ymin = min(values + ([0] if is_bar else []))
    ymax = max(values + ([0] if is_bar else []))
    if ymax == ymin:
        pad = max(abs(ymax) * .05, .5)
    else:
        pad = (ymax - ymin) * .08
    ymin -= pad
    ymax += pad
    def sy(y: float) -> float:
        return top + (ymax - y) * plot_h / (ymax - ymin)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{escape(str(chart["title"]), quote=True)}">',
           '<rect width="900" height="340" fill="#ffffff"/>',
           f'<text x="{left}" y="26" font-family="sans-serif" font-size="15" font-weight="600" fill="#222">{escape(str(chart["title"]))}</text>']
    for tick in range(5):
        value = ymin + (ymax - ymin) * tick / 4
        y = sy(value)
        out.append(f'<line x1="{left}" x2="{left+plot_w}" y1="{y:.2f}" y2="{y:.2f}" stroke="#e2e5e9"/>')
        out.append(f'<text x="{left-10}" y="{y+4:.2f}" text-anchor="end" font-family="monospace" font-size="11" fill="#444">{value:.4g}</text>')
    out.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#777"/>')
    if is_bar:
        span = plot_w / len(points)
        zero = sy(0)
        for i, p in enumerate(points):
            x = left + (i + .2) * span
            y = sy(float(p["y"]))
            out.append(f'<rect x="{x:.2f}" y="{min(y,zero):.2f}" width="{span*.6:.2f}" height="{max(abs(zero-y),.7):.2f}" fill="#526e88"/>')
            if len(points) <= 12 or i % max(1, len(points)//12) == 0:
                label = escape(str(p["label"])[:16])
                out.append(f'<text x="{left+(i+.5)*span:.2f}" y="{top+plot_h+20}" text-anchor="middle" font-family="monospace" font-size="10" fill="#444">{label}</text>')
    else:
        x_vals = [float(p["x"]) for p in points]
        x_min, x_max = min(x_vals), max(x_vals)
        def sx(x: float) -> float:
            return left + (x - x_min) / (x_max - x_min) * plot_w if x_max != x_min else left + plot_w/2
        coords = " ".join(f'{sx(x):.2f},{sy(y):.2f}' for x, y in zip(x_vals, values))
        out.append(f'<polyline points="{coords}" fill="none" stroke="#526e88" stroke-width="2" stroke-linejoin="round"/>')
        if len(points) <= 25:
            for x, y in zip(x_vals, values):
                out.append(f'<circle cx="{sx(x):.2f}" cy="{sy(y):.2f}" r="2.4" fill="#526e88"/>')
        for i in range(5):
            x = x_min + (x_max - x_min) * i / 4
            out.append(f'<text x="{sx(x):.2f}" y="{top+plot_h+20}" text-anchor="middle" font-family="monospace" font-size="11" fill="#444">{x:.4g}</text>')
    out.append(f'<text x="{left+plot_w/2}" y="{height-17}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#222">{escape(str(chart["x_label"]))}</text>')
    out.append(f'<text x="16" y="{top+plot_h/2}" text-anchor="middle" transform="rotate(-90 16 {top+plot_h/2})" font-family="sans-serif" font-size="12" fill="#222">{escape(str(chart["y_label"]))}</text>')
    out.append('</svg>')
    return "\n".join(out)
