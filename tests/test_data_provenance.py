from __future__ import annotations

import json
from pathlib import Path

from ml_lab import data
from ml_lab.application import execute_task


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def test_raw_dataset_is_valid_unrecorded_root(tmp_path: Path):
    source = _write(tmp_path / "raw.csv", "id,value\n1,10\n2,20\n")

    lineage = data.trace_lineage(source)

    assert lineage["schema"] == "ml-lab.data-lineage@1"
    assert lineage["status"] == "unrecorded_root"
    assert lineage["event_count"] == 0
    assert lineage["chain_valid"] is True
    assert lineage["authoritative"] is False
    assert lineage["root"]["sha256"] == lineage["target"]["sha256"]


def test_applied_transformation_persists_authoritative_event(tmp_path: Path):
    source = _write(tmp_path / "raw.csv", "id,value\n1,10\n2,20\n")
    derived = tmp_path / "derived.csv"
    recipe = {"name": "keep high", "operations": [{"type": "filter_rows", "column": "value", "operator": "ge", "value": 20}]}

    manifest = data.apply_transformation(source, recipe, output=derived)

    sidecar = data.provenance_sidecar_path(derived)
    assert sidecar.is_file()
    event = data.load_provenance_event(derived)
    assert event["schema"] == "ml-lab.transformation-event@1"
    assert event["authoritative"] is True
    assert event["source"]["sha256"] == data.inspect_file(source).sha256
    assert event["derived"]["sha256"] == data.inspect_file(derived).sha256
    assert event["recipe"]["snapshot"]["name"] == "keep high"
    assert event["changes"]["rows_before"] == 2
    assert event["changes"]["rows_after"] == 1
    assert manifest["provenance"]["event_id"] == event["event_id"]
    assert Path(manifest["provenance"]["path"]) == sidecar


def test_multistep_lineage_uses_explicit_parent_links(tmp_path: Path):
    raw = _write(tmp_path / "raw.csv", "id,value\n1,10\n2,20\n3,30\n")
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    data.apply_transformation(raw, {"operations": [{"type": "filter_rows", "column": "value", "operator": "ge", "value": 20}]}, output=first)
    first_event = data.load_provenance_event(first)
    data.apply_transformation(first, {"operations": [{"type": "rename_columns", "mapping": {"value": "score"}}]}, output=second)
    second_event = data.load_provenance_event(second)

    assert second_event["parent"]["event_id"] == first_event["event_id"]
    lineage = data.trace_lineage(second)
    assert lineage["authoritative"] is True
    assert lineage["chain_valid"] is True
    assert lineage["event_count"] == 2
    assert [event["event_id"] for event in lineage["events"]] == [first_event["event_id"], second_event["event_id"]]
    assert Path(lineage["root"]["path"]) == raw.resolve()
    assert Path(lineage["target"]["path"]) == second.resolve()


def test_tampered_derived_dataset_invalidates_chain(tmp_path: Path):
    raw = _write(tmp_path / "raw.csv", "id,value\n1,10\n2,20\n")
    derived = tmp_path / "derived.csv"
    data.apply_transformation(raw, {"operations": []}, output=derived)
    derived.write_text("id,value\n1,999\n", encoding="utf-8")

    lineage = data.trace_lineage(derived)

    assert lineage["event_count"] == 1
    assert lineage["chain_valid"] is False
    assert lineage["authoritative"] is False
    assert any("hash no longer matches" in warning for warning in lineage["warnings"])


def test_unlinked_sidecar_is_not_retroactively_promoted_to_parent(tmp_path: Path):
    raw = _write(tmp_path / "raw.csv", "id,value\n1,10\n2,20\n")
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    data.apply_transformation(raw, {"operations": []}, output=first)
    first_sidecar = data.provenance_sidecar_path(first)
    saved_event = first_sidecar.read_text(encoding="utf-8")
    first_sidecar.unlink()

    data.apply_transformation(first, {"operations": []}, output=second)
    second_event = data.load_provenance_event(second)
    assert second_event["parent"] is None

    # A matching sidecar appearing later must not rewrite the already-recorded history.
    first_sidecar.write_text(saved_event, encoding="utf-8")
    lineage = data.trace_lineage(second)
    assert lineage["event_count"] == 1
    assert Path(lineage["root"]["path"]) == first.resolve()


def test_lineage_application_task_and_save(tmp_path: Path):
    source = _write(tmp_path / "raw.csv", "x\n1\n")
    derived = tmp_path / "derived.csv"
    data.apply_transformation(source, {"operations": []}, output=derived)

    response = execute_task("data.lineage", {"path": str(derived)})
    assert response["task"] == "data.lineage"
    assert response["result"]["event_count"] == 1

    destination = tmp_path / "lineage.json"
    data.save_lineage(response["result"], destination)
    persisted = json.loads(destination.read_text(encoding="utf-8"))
    assert persisted["schema"] == "ml-lab.data-lineage@1"
