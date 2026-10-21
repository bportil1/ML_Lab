"""XML-C4: shared recipe is structural validation, never automatic extraction."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from ml_lab import data
from ml_lab.data.xml_collection_recipe import (XmlCollectionRecipeError, inspect_xml_collection_recipe,
                                               build_xml_collection_recipe, prepare_full_xml_collection_recipe,
                                               save_xml_collection_recipe, main)


def _fixtures(root: Path):
    (root / "a.xml").write_text('<rows><item id="1"><name>A</name><tag>one</tag><tag>two</tag></item></rows>')
    (root / "b.xml").write_text('<rows><item id="2"><name>B</name></item></rows>')
    (root / "c.xml").write_text('<other><item id="3"><name>C</name></item></other>')
    inventory = data.inventory_xml_collection(root)
    return inventory, data.compare_xml_collection(inventory)


def test_common_roots_field_catalog_and_variant_gate(tmp_path: Path):
    inventory, report = _fixtures(tmp_path)
    with pytest.raises(XmlCollectionRecipeError, match="acknowledgment"):
        inspect_xml_collection_recipe(inventory, report, ["a.xml", "b.xml"])
    catalog = inspect_xml_collection_recipe(inventory, report, ["a.xml", "b.xml"],
                                                acknowledge_variants=True, record_root_canonical_path="/rows/item")
    assert "/rows/item" in {row["canonical_path"] for row in catalog["record_root_options"]}
    assert catalog["source_fingerprints"]["a.xml"].startswith("sha256:")
    fields = {item["field_id"]: item for item in catalog["fields"]}
    assert fields["text:/rows/item/name"]["missing_in"] == []
    assert fields["text:/rows/item/tag"]["missing_in"] == ["b.xml"]
    assert fields["text:/rows/item/tag"]["repeated_in"] == ["a.xml"]
    with pytest.raises(XmlCollectionRecipeError, match="record root"):
        inspect_xml_collection_recipe(inventory, report, ["a.xml", "b.xml"],
                                           acknowledge_variants=True, record_root_canonical_path="/rows/item/tag")


def test_missing_and_repeated_rules_are_explicit(tmp_path: Path):
    inventory, report = _fixtures(tmp_path)
    kwargs = dict(acknowledge_variants=True, selected_paths=["a.xml", "b.xml"],
                  record_root_canonical_path="/rows/item",
                  selected_field_ids=["text:/rows/item/name", "text:/rows/item/tag"])
    blocked = build_xml_collection_recipe(inventory, report, **kwargs)
    assert blocked["status"] == "blocked"
    assert blocked["required_repeated_branches"] == ["/rows/item/tag"]
    assert not blocked["merge_authorized"] and not blocked["csv_generated"]
    still_blocked = build_xml_collection_recipe(
        inventory, report, allow_missing_fields=True, **kwargs)
    assert still_blocked["status"] == "blocked"
    rule = {"branch_canonical_path": "/rows/item/tag", "strategy": "join", "join_delimiter": ","}
    ready = build_xml_collection_recipe(inventory, report, allow_missing_fields=True,
                                             rules=[rule], **kwargs)
    assert ready["status"] == "ready_for_collection_preview"
    assert ready["structural_recipe_validated"] and not ready["merge_authorized"]
    assert ready["selected_sources"][1]["missing_field_ids"] == ["text:/rows/item/tag"]
    destination = tmp_path / "out" / "recipe.json"
    assert save_xml_collection_recipe(ready, destination) == destination
    assert json.loads(destination.read_text())["rules"] == [rule]
    with pytest.raises(XmlCollectionRecipeError, match="only validated"):
        save_xml_collection_recipe(blocked, tmp_path / "invalid.json")
    assert not list(tmp_path.glob("*.csv"))


def test_invalid_or_stale_evidence_fails_closed(tmp_path: Path):
    inventory, report = _fixtures(tmp_path)
    with pytest.raises(XmlCollectionRecipeError, match="blocked"):
        inspect_xml_collection_recipe(inventory, report, ["a.xml", "c.xml"], acknowledge_variants=True)
    altered = {**report, "root_path": "/another/directory"}
    with pytest.raises(XmlCollectionRecipeError, match="different XML directory"):
        inspect_xml_collection_recipe(inventory, altered, ["a.xml", "b.xml"], acknowledge_variants=True)
    (tmp_path / "b.xml").write_text('<rows><item id="2"><name>changed</name></item></rows>')
    with pytest.raises(XmlCollectionRecipeError, match="changed"):
        inspect_xml_collection_recipe(inventory, report, ["a.xml", "b.xml"], acknowledge_variants=True)


def test_unknown_fields_and_unused_rules_fail(tmp_path: Path):
    inventory, report = _fixtures(tmp_path)
    kwargs = dict(acknowledge_variants=True, allow_missing_fields=True)
    with pytest.raises(XmlCollectionRecipeError, match="unknown field"):
        build_xml_collection_recipe(inventory, report, ["a.xml", "b.xml"],
                                         "/rows/item", ["bogus"], **kwargs)
    blocked = build_xml_collection_recipe(
        inventory, report, ["a.xml", "b.xml"], "/rows/item", ["text:/rows/item/name"],
        rules=[{"branch_canonical_path": "/rows/item/unrelated", "strategy": "first"}], **kwargs)
    assert "Unused or unknown" in " ".join(blocked["blocking_reasons"])
    with pytest.raises(XmlCollectionRecipeError, match="duplicate"):
        build_xml_collection_recipe(inventory, report, ["a.xml", "b.xml"],
                                         "/rows/item", ["text:/rows/item/name"] * 2, **kwargs)


def test_ui_flow_and_json_export(tmp_path: Path):
    pytest.importorskip("flask")
    from ml_lab.ui import create_app

    _fixtures(tmp_path)
    client = create_app(config={"TESTING": True}).test_client()
    url = "/data/xml/collection/recipe"
    assert client.get(url).status_code == 200
    base = {"directory": str(tmp_path), "recursive": "1", "selected": ["a.xml", "b.xml"],
            "acknowledge_variants": "1", "max_files": "40"}
    loaded = client.post(url, data={**base, "mode": "inspect"})
    assert loaded.status_code == 200
    assert b"/rows/item" in loaded.data
    fields = client.post(url, data={**base, "mode": "fields", "record_root": "/rows/item"})
    assert fields.status_code == 200
    assert b"text:/rows/item/name" in fields.data
    chosen = {**base, "record_root": "/rows/item", "field": ["text:/rows/item/name", "text:/rows/item/tag"],
              "allow_missing_fields": "1", "branch_path": ["/rows/item/tag"]}
    branches = client.post(url, data={**chosen, "mode": "branches"})
    assert b"Repeated-branch strategies" in branches.data
    assert b"blocked" in branches.data
    valid = {**chosen, "strategy::/rows/item/tag": "join", "join_delimiter::/rows/item/tag": ","}
    exported = client.post(url, data={**valid, "mode": "export"})
    assert exported.status_code == 200
    assert exported.get_json()["status"] == "ready_for_collection_preview"
    assert not exported.get_json()["csv_generated"]
    assert exported.mimetype == "application/json"
    assert "attachment" in exported.headers["Content-Disposition"]
    assert client.post(url, data={**chosen, "mode": "export"}).status_code == 400


def test_cli_inspect_and_validate(tmp_path: Path, capsys):
    _fixtures(tmp_path)
    command = [str(tmp_path),
               "--source", "a.xml", "--source", "b.xml", "--acknowledge-variants"]
    assert main(command + ["--inspect", "--record-root", "/rows/item"]) == 0
    assert "text:/rows/item/tag" in capsys.readouterr().out
    rulefile = tmp_path / "rules.json"
    rulefile.write_text(json.dumps([{"branch_canonical_path": "/rows/item/tag", "strategy": "join", "join_delimiter": ","}]))
    output = tmp_path / "recipe.json"
    assert main(command + ["--record-root", "/rows/item", "--field", "text:/rows/item/name",
                           "--field", "text:/rows/item/tag", "--allow-missing-fields",
                           "--rules-json", str(rulefile), "--output", str(output)]) == 0
    assert json.loads(output.read_text())["structural_recipe_validated"] is True


def test_prepare_full_group_recipe_selects_all_fields_and_logical_records(tmp_path: Path):
    (tmp_path / "a.xml").write_text(
        '<rows><item id="1"><name>A</name><tag>one</tag><tag>two</tag></item></rows>',
        encoding="utf-8",
    )
    (tmp_path / "b.xml").write_text(
        '<rows><item id="2"><name>B</name></item></rows>', encoding="utf-8",
    )
    inventory = data.inventory_xml_collection(tmp_path)
    report = data.compare_xml_collection(inventory)
    catalog, recipe = prepare_full_xml_collection_recipe(inventory, report, ["a.xml", "b.xml"])
    assert recipe["status"] == "ready_for_collection_preview"
    assert recipe["record_root_canonical_path"] == "/rows/item"
    assert set(recipe["selected_field_ids"]) == {
        "attribute:/rows/item@id", "text:/rows/item/name", "text:/rows/item/tag",
    }
    assert recipe["allow_missing_fields"] is True
    assert recipe["rules"] == [{"branch_canonical_path": "/rows/item/tag", "strategy": "keep_nested"}]
    assert catalog["record_root_canonical_path"] == "/rows/item"

    from ml_lab.data.xml_collection_export import preview_xml_collection_export
    preview = preview_xml_collection_export(recipe)
    assert len(preview["sources"]) == 2
    assert len(preview["rows"]) == 2
    assert {row["__source_file"] for row in preview["rows"]} == {"a.xml", "b.xml"}
    assert preview["rows"][0]["tag"] == [{"tag": "one"}, {"tag": "two"}]


def test_prepare_full_group_recipe_preserves_multiple_records_per_source(tmp_path: Path):
    (tmp_path / "a.xml").write_text(
        '<rows><item id="1"><name>A</name></item><item id="2"><name>B</name></item></rows>',
        encoding="utf-8",
    )
    (tmp_path / "b.xml").write_text(
        '<rows><item id="3"><name>C</name></item><item id="4"><name>D</name></item></rows>',
        encoding="utf-8",
    )
    inventory = data.inventory_xml_collection(tmp_path)
    report = data.compare_xml_collection(inventory)
    _, recipe = prepare_full_xml_collection_recipe(inventory, report, ["a.xml", "b.xml"])
    assert recipe["record_root_canonical_path"] == "/rows/item"
    from ml_lab.data.xml_collection_export import preview_xml_collection_export
    preview = preview_xml_collection_export(recipe)
    assert len(preview["rows"]) == 4
