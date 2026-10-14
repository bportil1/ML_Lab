"""XML-C5: explicit, traceable multi-XML consolidation without implicit merges."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ml_lab.data.xml_collection import inventory_xml_collection
from ml_lab.data.xml_collection_export import (
    XmlCollectionExportError, main, materialize_xml_collection,
    preview_xml_collection_export,
)
from ml_lab.data.xml_collection_recipe import build_xml_collection_recipe, save_xml_collection_recipe
from ml_lab.data.xml_similarity import compare_xml_collection


def _ready(root: Path, *, with_child: bool = False) -> dict:
    (root / 'a.xml').write_text('<rows><item id="1"><name>Alpha</name><tag>red</tag><tag>blue</tag></item></rows>', encoding='utf-8')
    (root / 'b.xml').write_text('<rows><item id="2"><name>Beta</name></item></rows>', encoding='utf-8')
    inventory = inventory_xml_collection(root)
    report = compare_xml_collection(inventory)
    rules = [{'branch_canonical_path': '/rows/item/tag',
              'strategy': 'separate_table' if with_child else 'join',
              'separate_table_name': 'tags' if with_child else '',
              'join_delimiter': '|'}]
    recipe = build_xml_collection_recipe(
        inventory, report, ['a.xml', 'b.xml'], '/rows/item',
        ['text:/rows/item/name', 'text:/rows/item/tag'],
        rules=rules, allow_missing_fields=True, acknowledge_variants=True,
    )
    assert recipe['status'] == 'ready_for_collection_preview'
    return recipe


def test_two_file_consolidation_alignment_and_source_lineage(tmp_path: Path):
    recipe = _ready(tmp_path)
    sample = preview_xml_collection_export(recipe)
    assert sample['no_files_written'] and sample['rows'][0]['__source_file'] == 'a.xml'
    assert not list(tmp_path.glob('*.csv'))
    out = tmp_path / 'combined.csv'
    with pytest.raises(XmlCollectionExportError, match='Explicit confirmation'):
        materialize_xml_collection(recipe, out)
    manifest = materialize_xml_collection(recipe, out, confirm=True)
    assert manifest['schema'] == 'ml-lab.xml-collection-materialization@1'
    assert manifest['provenance']['status'] == 'not_recorded'
    assert manifest['main_dataset']['row_count'] == 2
    with out.open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        assert reader.fieldnames == ['__source_file', '__source_row', 'name', 'tag']
        rows = list(reader)
    assert rows == [{'__source_file': 'a.xml', '__source_row': '1', 'name': 'Alpha', 'tag': 'red|blue'},
                    {'__source_file': 'b.xml', '__source_row': '1', 'name': 'Beta', 'tag': ''}]
    assert [x['relative_path'] for x in manifest['sources']] == ['a.xml', 'b.xml']
    assert (tmp_path / 'combined.csv.collection.json').exists()
    assert (tmp_path / 'a.xml').read_text().startswith('<rows>')


def test_child_tables_retain_source_and_parent_index(tmp_path: Path):
    recipe = _ready(tmp_path, with_child=True)
    report = materialize_xml_collection(recipe, tmp_path / 'all.tsv', confirm=True)
    assert len(report['child_datasets']) == 1
    child = tmp_path / 'all__tags.tsv'
    with child.open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    assert len(rows) == 2
    assert all(row['__source_file'] == 'a.xml' and row['__parent_record'] == '1' for row in rows)


def test_reject_stale_source_and_invalid_recipe(tmp_path: Path):
    recipe = _ready(tmp_path)
    bad = {**recipe, 'structural_recipe_validated': False}
    with pytest.raises(XmlCollectionExportError, match='validated'):
        preview_xml_collection_export(bad)
    (tmp_path / 'a.xml').write_text('<rows><item><name>tampered</name></item></rows>', encoding='utf-8')
    with pytest.raises(XmlCollectionExportError, match='changed'):
        materialize_xml_collection(recipe, tmp_path / 'out.csv', confirm=True)
    assert not (tmp_path / 'out.csv').exists()


def test_output_collision_is_not_silently_overwritten(tmp_path: Path):
    recipe = _ready(tmp_path)
    target = tmp_path / 'all.csv'
    target.write_text('keep_me', encoding='utf-8')
    with pytest.raises(FileExistsError):
        materialize_xml_collection(recipe, target, confirm=True)
    assert target.read_text() == 'keep_me'
    data = materialize_xml_collection(recipe, target, confirm=True, overwrite=True)
    assert data['main_dataset']['row_count'] == 2
    assert target.read_text() != 'keep_me'


def test_symlink_substitution_and_unsupported_extension(tmp_path: Path):
    recipe = _ready(tmp_path)
    with pytest.raises(XmlCollectionExportError, match='csv or .tsv'):
        materialize_xml_collection(recipe, tmp_path / 'bad.txt', confirm=True)
    target = tmp_path / 'links.csv'
    target.symlink_to(tmp_path / 'a.xml')
    with pytest.raises(XmlCollectionExportError, match='symlink'):
        materialize_xml_collection(recipe, target, confirm=True)


def test_hard_output_limit_prevents_publication(tmp_path: Path):
    recipe = _ready(tmp_path)
    with pytest.raises(XmlCollectionExportError, match='row limit'):
        materialize_xml_collection(recipe, tmp_path / 'no.csv', confirm=True, max_total_rows=1)
    assert not list(tmp_path.glob('*.csv'))


def test_module_cli_preview_and_export(tmp_path: Path, capsys):
    recipe = _ready(tmp_path)
    recipe_json = save_xml_collection_recipe(recipe, tmp_path / 'recipe.json')
    assert main(['--recipe', str(recipe_json), '--preview']) == 0
    assert 'no_files_written' in capsys.readouterr().out
    assert main(['--recipe', str(recipe_json), '--output', str(tmp_path / 'combo.csv'), '--confirm']) == 0
    assert (tmp_path / 'combo.csv').exists()


def test_ui_route_requires_explicit_confirmation(tmp_path: Path):
    pytest.importorskip('flask')
    from ml_lab.ui import create_app
    recipe = _ready(tmp_path)
    client = create_app(config={'TESTING': True}).test_client()
    path = '/data/xml/collection/export'
    assert client.get(path).status_code == 200
    data = {'recipe_json': json.dumps(recipe), 'output': str(tmp_path / 'combo.csv')}
    preview = client.post(path, data={**data, 'mode': 'preview'})
    assert preview.status_code == 200 and b'Alpha' in preview.data
    assert client.post(path, data={**data, 'mode': 'export'}).status_code == 400
    assert client.post(path, data={**data, 'mode': 'export', 'confirm': '1'}).status_code == 200
    assert (tmp_path / 'combo.csv').exists()
