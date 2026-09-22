from __future__ import annotations

import json
from pathlib import Path

from ml_lab.cli import main


def test_data_inspect_cli_writes_inventory(tmp_path: Path, capsys):
    source = tmp_path / "table.csv"
    source.write_text("a,b\n1,2\n3,4\n", encoding="utf-8")
    output = tmp_path / "inventory.json"

    status = main(["data", "inspect", str(source), "--output", str(output)])
    captured = capsys.readouterr().out
    assert status == 0
    assert "Inventory written to" in captured
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["summary"]["parsed_file_count"] == 1


def test_data_profile_cli_writes_profile(tmp_path: Path, capsys):
    source = tmp_path / "table.csv"
    source.write_text("x,y\n1,2\n2,4\n3,6\n", encoding="utf-8")
    output = tmp_path / "profile.json"

    status = main(["data", "profile", str(source), "--relationship-rows", "0", "--output", str(output)])
    captured = capsys.readouterr().out
    assert status == 0
    assert "Profile written to" in captured
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "ml-lab.data-profile-collection@1"
    assert payload["summary"]["profile_count"] == 1


def test_data_transform_cli_preview_and_apply(tmp_path: Path, capsys):
    source = tmp_path / "source.csv"
    source.write_text("name,value\n A ,1\n B ,2\n", encoding="utf-8")
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps({"operations": [{"type": "clean_strings", "columns": ["name"]}]}), encoding="utf-8")

    status = main(["data", "transform", str(source), "--recipe", str(recipe_path), "--preview"])
    captured = capsys.readouterr().out
    assert status == 0
    assert "ml-lab.transformation-preview@1" in captured

    output = tmp_path / "derived.csv"
    status = main(["data", "transform", str(source), "--recipe", str(recipe_path), "--output", str(output)])
    captured = capsys.readouterr().out
    assert status == 0
    assert "Derived dataset written to" in captured
    assert output.is_file()


def test_data_compare_cli(tmp_path, capsys):
    first = tmp_path / "one.csv"
    second = tmp_path / "two.csv"
    output = tmp_path / "comparison.json"
    first.write_text("id,x\n1,10\n2,20\n", encoding="utf-8")
    second.write_text("id,x\n1,10\n2,21\n", encoding="utf-8")
    assert main(["data", "compare", str(first), str(second), "--output", str(output)]) == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "ml-lab.dataset-comparison-collection@1"
    assert payload["summary"]["compared_pair_count"] == 1
    assert "pairs=1/1" in capsys.readouterr().out


def test_data_lineage_cli_writes_authoritative_lineage(tmp_path: Path, capsys):
    from ml_lab import data

    source = tmp_path / "source.csv"
    source.write_text("id,value\n1,10\n2,20\n", encoding="utf-8")
    derived = tmp_path / "derived.csv"
    data.apply_transformation(
        source,
        {"operations": [{"type": "filter_rows", "column": "value", "operator": "ge", "value": 20}]},
        output=derived,
    )
    output = tmp_path / "lineage.json"

    status = main(["data", "lineage", str(derived), "--output", str(output)])
    captured = capsys.readouterr().out

    assert status == 0
    assert "events=1" in captured
    assert "authoritative=True" in captured
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema"] == "ml-lab.data-lineage@1"
    assert payload["event_count"] == 1
