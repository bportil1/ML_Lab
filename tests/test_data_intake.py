from __future__ import annotations

import json
from pathlib import Path

from ml_lab import data


def test_csv_inventory_detects_shape_header_delimiter_and_exported_index(tmp_path: Path):
    source = tmp_path / "sample.csv"
    source.write_text(
        "Unnamed: 0,feature_a,feature_b\n"
        "0,1.5,alpha\n"
        "1,2.5,beta\n"
        "2,3.5,gamma\n",
        encoding="utf-8",
    )

    inventory = data.inspect_paths([tmp_path])
    assert inventory.discovered_file_count == 1
    record = inventory.files[0]
    assert record.parse_status == "parsed"
    assert record.delimiter == ","
    assert record.has_header is True
    assert record.row_count == 3
    assert record.column_count == 3
    assert record.columns == ("Unnamed: 0", "feature_a", "feature_b")
    assert record.likely_exported_index_columns == ("Unnamed: 0",)
    assert len(record.sha256) == 64


def test_tsv_and_malformed_width_rows_are_reported_without_mutation(tmp_path: Path):
    source = tmp_path / "messy.tsv"
    original = "name\tvalue\nalpha\t1\nbeta\t2\textra\ngamma\t3\n"
    source.write_text(original, encoding="utf-8")

    record = data.inspect_file(source)
    assert record.delimiter == "\t"
    assert record.row_count == 3
    assert record.malformed_row_count == 1
    assert record.parse_status == "partial"
    assert record.malformed_rows[0].reason == "column_count_mismatch"
    assert source.read_text(encoding="utf-8") == original


def test_inventory_recurses_skips_hidden_and_records_unsupported_files(tmp_path: Path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "data.csv").write_text("a,b\n1,2\n2,3\n", encoding="utf-8")
    (nested / "notes.md").write_text("hello", encoding="utf-8")
    hidden = tmp_path / ".hidden"
    hidden.mkdir()
    (hidden / "secret.csv").write_text("a,b\n9,9\n", encoding="utf-8")

    inventory = data.inspect_paths([tmp_path])
    assert inventory.discovered_file_count == 2
    assert inventory.supported_file_count == 1
    assert inventory.unsupported_file_count == 1
    unsupported = next(record for record in inventory.files if not record.supported)
    assert unsupported.parse_status == "unsupported"
    assert unsupported.sha256 == ""

    with_hidden = data.inspect_paths([tmp_path], include_hidden=True)
    assert with_hidden.discovered_file_count == 3


def test_inventory_records_missing_path_as_top_level_warning(tmp_path: Path):
    inventory = data.inspect_paths([tmp_path / "does-not-exist"])
    assert inventory.discovered_file_count == 0
    assert "path does not exist" in inventory.warnings[0]


def test_inventory_reporting_writes_schema_json(tmp_path: Path):
    source = tmp_path / "sample.csv"
    source.write_text("x,y\n1,2\n2,3\n", encoding="utf-8")
    inventory = data.inspect_paths([source])

    output = data.save_inventory(inventory, tmp_path / "report")
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert output.name == "inventory.json"
    assert payload["schema"] == "ml-lab.data-inventory@1"
    assert payload["summary"]["parsed_file_count"] == 1
