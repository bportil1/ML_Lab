from __future__ import annotations

from ml_lab.ui import capability_records, get_capability


def test_ui_capabilities_are_available_without_importing_flask():
    records = capability_records()
    by_id = {item["id"]: item for item in records}
    assert "classification" in by_id
    assert by_id["classification"]["task"] == "classification"
    assert "experimental" not in by_id


def test_experimental_capability_is_explicit_opt_in():
    by_id = {item["id"]: item for item in capability_records(enable_experimental=True)}
    assert by_id["experimental"]["effective_status"] == "experimental"


def test_data_lab_is_visible_as_next_planned_workspace():
    data = get_capability("data")
    assert data["status"] == "planned"
    assert data["available"] is False
