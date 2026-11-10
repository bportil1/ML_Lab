"""Persistent opt-in typed-preview artifacts: identity, isolation, and tamper tests."""
import hashlib
import pytest

from ml_lab.data.xml_typed_artifacts import persist_typed_preview, read_typed_preview
from ml_lab.data.xml_typed_records import PREVIEW_SCHEMA


def _fixture():
    recipe = {"schema": "ml-lab.xml-typed-recipe@1", "version": 1,
              "record_root_canonical_path": "/root/row", "variables": [],
              "linked_child_paths": [], "encoding": "none", "legacy_extraction_modified": False}
    preview = {"schema": PREVIEW_SCHEMA, "source_sha256": "f" * 64,
               "source_fingerprint": "sha256:f", "record_root_canonical_path": "/root/row",
               "encoding": "none", "records": [{"record_index": 0, "variables": [
                   {"canonical_path": "/root/row/x", "present": False, "items": [], "count": 0},
                   {"canonical_path": "/root/row/y", "present": True, "items": [
                       {"lineage": {"canonical_path": "/root/row/y", "occurrence_address": [0, 0]},
                        "value": {"kind": "scalar", "present": True, "value": ""}}], "count": 1}]}],
               "preview_record_count": 1, "source_record_count": 2, "truncated": True,
               "registered_dataset": False}
    return preview, recipe


def test_persist_roundtrip_content_addressed_and_idempotent(tmp_path):
    preview, recipe = _fixture()
    result = persist_typed_preview(tmp_path, preview, recipe=recipe)
    assert result["truncated"] and not result["registered_dataset"]
    assert persist_typed_preview(tmp_path, preview, recipe=recipe) == result
    saved = read_typed_preview(tmp_path, result["artifact_id"])
    assert saved["preview"] == preview
    assert saved["preview"]["records"][0]["variables"][0]["present"] is False
    assert saved["preview"]["records"][0]["variables"][1]["items"][0]["value"]["value"] == ""
    assert saved["recipe"] == recipe


def test_artifact_is_project_local(tmp_path):
    preview, recipe = _fixture()
    result = persist_typed_preview(tmp_path / "a", preview, recipe=recipe)
    with pytest.raises(FileNotFoundError):
        read_typed_preview(tmp_path / "b", result["artifact_id"])


def test_tamper_detection_and_path_rejection(tmp_path):
    preview, recipe = _fixture()
    result = persist_typed_preview(tmp_path, preview, recipe=recipe)
    path = tmp_path / result["relative_path"]
    path.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        read_typed_preview(tmp_path, result["artifact_id"])
    with pytest.raises(ValueError, match="differs"):
        persist_typed_preview(tmp_path, preview, recipe=recipe)
    with pytest.raises(ValueError, match="artifact ID"):
        read_typed_preview(tmp_path, "../outside")


def test_reject_unbounded_or_encoded_payload(tmp_path):
    preview, recipe = _fixture()
    preview["registered_dataset"] = True
    with pytest.raises(ValueError):
        persist_typed_preview(tmp_path, preview, recipe=recipe)
    preview["registered_dataset"] = False
    preview["preview_record_count"] = 42
    with pytest.raises(ValueError):
        persist_typed_preview(tmp_path, preview, recipe=recipe)
    preview["preview_record_count"] = 1
    recipe["encoding"] = "numeric"
    with pytest.raises(ValueError):
        persist_typed_preview(tmp_path, preview, recipe=recipe)
