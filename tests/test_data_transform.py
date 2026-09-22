from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ml_lab import data
from ml_lab.application import execute_task


def test_preview_transformation_is_non_destructive_and_reports_diagnostics(tmp_path: Path):
    source = tmp_path / "source.csv"
    original = "id,trial,score,status,label\n1,1,10,NA,aug_3\n2,x,100, ok ,aug_6\n2,x,100, ok ,aug_6\n"
    source.write_text(original, encoding="utf-8")
    recipe = {
        "name": "checkpoint cleanup",
        "operations": [
            {"type": "coerce_types", "mapping": {"trial": "integer"}, "errors": "coerce"},
            {"type": "sentinel_to_missing", "columns": ["status"], "values": ["NA"]},
            {"type": "clean_strings", "columns": ["status"], "strip": True},
            {"type": "derive", "method": "regex_extract", "source": "label", "target": "augmentation", "pattern": r"aug_(\d+)", "group": 0},
            {"type": "drop_duplicates", "columns": ["id", "score", "label"], "keep": "first"},
            {"type": "outliers", "columns": ["score"], "method": "clip_iqr", "iqr_multiplier": 1.5},
        ],
    }

    preview = data.preview_transformation(source, recipe, preview_rows=10)

    assert preview["schema"] == "ml-lab.transformation-preview@1"
    assert preview["summary"]["rows_before"] == 3
    assert preview["summary"]["rows_after"] == 2
    assert preview["summary"]["coercion_failures"] == 2
    assert "augmentation" in preview["preview"]["columns"]
    assert source.read_text(encoding="utf-8") == original


def test_apply_transformation_writes_derived_recipe_and_manifest_without_touching_source(tmp_path: Path):
    source = tmp_path / "source.csv"
    source.write_text("id,value,group\n1,1,A\n2,2,B\n3,3,A\n", encoding="utf-8")
    before_hash = data.inspect_file(source).sha256
    output = tmp_path / "derived.csv"
    recipe = {
        "operations": [
            {"type": "filter_rows", "column": "value", "operator": "ge", "value": 2},
            {"type": "encode_categorical", "columns": ["group"], "method": "one_hot"},
            {"type": "scale", "columns": ["value"], "method": "minmax"},
        ]
    }

    manifest = data.apply_transformation(source, recipe, output=output)

    assert manifest["schema"] == "ml-lab.derived-dataset@1"
    assert output.is_file()
    assert Path(manifest["recipe_path"]).is_file()
    assert Path(manifest["manifest_path"]).is_file()
    assert Path(manifest["provenance"]["path"]).is_file()
    assert data.inspect_file(source).sha256 == before_hash
    derived = pd.read_csv(output)
    assert list(derived["value"]) == [0.0, 1.0]
    assert {"group_A", "group_B"}.issubset(derived.columns)
    persisted = json.loads(Path(manifest["manifest_path"]).read_text(encoding="utf-8"))
    assert persisted["source"]["sha256"] == before_hash
    assert persisted["derived"]["sha256"] == manifest["derived"]["sha256"]
    assert persisted["provenance"]["event_id"] == manifest["provenance"]["event_id"]


def test_apply_refuses_raw_source_overwrite_and_existing_derived_without_explicit_permission(tmp_path: Path):
    source = tmp_path / "source.csv"
    source.write_text("x\n1\n", encoding="utf-8")
    recipe = {"operations": []}

    with pytest.raises(ValueError, match="raw source"):
        data.apply_transformation(source, recipe, output=source, overwrite=True)

    output = tmp_path / "derived.csv"
    output.write_text("old\nvalue\n", encoding="utf-8")
    with pytest.raises(FileExistsError):
        data.apply_transformation(source, recipe, output=output)

    data.apply_transformation(source, recipe, output=output, overwrite=True)
    assert output.read_text(encoding="utf-8").startswith("x")


def test_malformed_rows_require_explicit_recipe_permission(tmp_path: Path):
    source = tmp_path / "bad.csv"
    source.write_text("a,b\n1,2\n3,4,5\n6,7\n", encoding="utf-8")

    with pytest.raises(ValueError, match="malformed"):
        data.preview_transformation(source, {"operations": []})

    result = data.preview_transformation(source, {"allow_malformed_rows": True, "operations": []})
    assert result["summary"]["rows_after"] == 2
    assert result["warnings"]


def test_application_tasks_share_transform_contract(tmp_path: Path):
    source = tmp_path / "source.csv"
    source.write_text("name,value\n A ,1\n B ,2\n", encoding="utf-8")
    recipe = {"operations": [{"type": "clean_strings", "columns": ["name"], "strip": True}]}

    preview = execute_task("data.transform.preview", {"path": str(source), "recipe": recipe})
    assert preview["task"] == "data.transform.preview"
    assert preview["result"]["preview"]["rows"][0][0] == "A"

    output = tmp_path / "out.csv"
    applied = execute_task("data.transform.apply", {"path": str(source), "recipe": recipe, "output": str(output)})
    assert applied["task"] == "data.transform.apply"
    assert Path(applied["result"]["derived"]["path"]) == output.resolve()


def test_join_melt_and_pivot_are_available_through_recipe_contract(tmp_path: Path):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text("id,a,b\n1,10,20\n2,30,40\n", encoding="utf-8")
    right.write_text("id,label\n1,X\n2,Y\n", encoding="utf-8")

    joined = data.preview_transformation(left, {"operations": [{"type": "join", "right_path": str(right), "on": "id", "how": "left"}]})
    assert "label" in joined["preview"]["columns"]

    melted = data.preview_transformation(left, {"operations": [{"type": "melt", "id_vars": ["id"], "value_vars": ["a", "b"]}]})
    assert melted["summary"]["rows_after"] == 4

    long_source = tmp_path / "long.csv"
    long_source.write_text("id,metric,value\n1,a,10\n1,b,20\n2,a,30\n2,b,40\n", encoding="utf-8")
    pivoted = data.preview_transformation(long_source, {"operations": [{"type": "pivot", "index": "id", "columns": "metric", "values": "value", "aggfunc": "first"}]})
    assert {"a", "b"}.issubset(pivoted["preview"]["columns"])
