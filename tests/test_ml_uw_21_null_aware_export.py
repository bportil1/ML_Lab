"""ML-UW-21: verified C5/C7 lossless nulls, Comments and default dataset features."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ml_lab.data.xml_collection import inventory_xml_collection
from ml_lab.data.xml_collection_export import materialize_xml_collection, preview_xml_collection_export
from ml_lab.data.xml_collection_scale import materialize_xml_collection_scaled
from ml_lab.data.xml_collection_recipe import build_xml_collection_recipe
from ml_lab.data.xml_similarity import compare_xml_collection
from ml_lab.data.xml_null_export import (
    NULL_TOKEN, EMPTY_TOKEN, encode_value, decode_value,
    read_null_aware_collection_table, verify_null_aware_collection_table,
)
from ml_lab.data.provenance import read_dataset_frame
from ml_lab.data.table import read_table_page
from ml_lab.data.profiling import _load_frame
from ml_lab.data.intake import inspect_file
from ml_lab.experiments import ExperimentStore
from ml_lab.experiments.datasets import describe_selection


def _recipe(root: Path):
    (root / 'a.xml').write_text(
        '<rows xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<item code="7"><name>Alpha</name><value xsi:nil="true"/></item>'
        '<item><name>Beta</name><value/></item>'
        '<item><name>\\N</name><value>\\E</value></item>'
        '</rows>', encoding='utf-8')
    (root / 'b.xml').write_text(
        '<rows><item><name>Gamma</name></item><item><name>Delta</name></item></rows>', encoding='utf-8')
    (root / 'c.xml').write_text(
        '<rows><item><other>unused</other></item></rows>', encoding='utf-8')
    inventory = inventory_xml_collection(root)
    report = compare_xml_collection(inventory)
    recipe = build_xml_collection_recipe(
        inventory, report, ['a.xml', 'b.xml', 'c.xml'], '/rows/item',
        ['text:/rows/item/name', 'text:/rows/item/value', 'attribute:/rows/item@code'],
        acknowledge_variants=True,
    )
    assert recipe['status'] == 'ready_for_collection_preview'
    return recipe


@pytest.mark.parametrize('suffix,scaled', [('.csv', False), ('.tsv', False), ('.csv', True), ('.tsv', True)])
def test_null_and_empty_roundtrip_with_comments(tmp_path: Path, suffix: str, scaled: bool):
    recipe = _recipe(tmp_path)
    path = tmp_path / ('combined' + suffix)
    exporter = materialize_xml_collection_scaled if scaled else materialize_xml_collection
    manifest = exporter(recipe, path, confirm=True)
    assert manifest['main_dataset']['columns'][-1] == 'Comments'
    assert manifest['main_dataset']['null_encoding']['schema'] == 'ml-lab.xml-null-delimited@1'
    assert verify_null_aware_collection_table(path)['row_count'] == 6
    with path.open('r', encoding='utf-8', newline='') as stream:
        encoded = list(csv.DictReader(stream, delimiter='\t' if suffix == '.tsv' else ','))
    assert encoded[0]['value'] == NULL_TOKEN
    assert encoded[1]['value'] == EMPTY_TOKEN
    assert encoded[2]['name'] == '\\' + NULL_TOKEN
    assert encoded[2]['value'] == '\\' + EMPTY_TOKEN
    assert encoded[3]['value'] == NULL_TOKEN
    assert json.loads(encoded[0]['Comments']) == {'value': 'explicit_xsi_nil'}
    assert json.loads(encoded[1]['Comments']) == {'@code': 'attribute_absent'}
    assert json.loads(encoded[3]['Comments']) == {'@code': 'attribute_absent', 'value': 'path_absent'}
    assert json.loads(encoded[5]['Comments']) == {
        '@code': 'attribute_absent', 'name': 'path_absent', 'value': 'path_absent',
    }
    decoded = read_null_aware_collection_table(path)
    assert decoded[0]['value'] is None and decoded[1]['value'] == ''
    assert decoded[2]['name'] == NULL_TOKEN and decoded[2]['value'] == EMPTY_TOKEN
    assert decoded[5]['name'] is None
    # The main Data Lab read path also honors the recorded null encoding.
    frame = read_dataset_frame(path)
    assert frame.loc[0, 'value'] is None
    assert frame.loc[1, 'value'] == ''
    assert frame.loc[2, 'name'] == NULL_TOKEN
    page = read_table_page(path)
    assert page['rows'][0][3] is None and page['rows'][1][3] == ''


def test_profiler_and_default_feature_selection_exclude_comments(tmp_path: Path):
    from ml_lab.experiments.store import ExperimentStore

    root = tmp_path / 'project'
    root.mkdir()
    recipe = _recipe(root)
    target = root / 'dataset.csv'
    materialize_xml_collection(recipe, target, confirm=True)
    record = inspect_file(target)
    frame, count, bad = _load_frame(record, max_rows=None, random_state=1)
    assert count == 6 and bad == 0
    assert frame.loc[0, 'value'] is None
    assert frame.loc[1, 'value'] == ''
    # The experiment store is project-scoped; Comments is opt-in, not a feature.
    store = ExperimentStore(root)
    selected = describe_selection(store, target)
    assert 'Comments' in selected['column_names']
    assert 'Comments' not in selected['feature_columns']
    assert 'Comments' in describe_selection(store, target, feature_columns=['name', 'Comments'])['feature_columns']


def test_fails_closed_for_tampered_encoded_export(tmp_path: Path):
    recipe = _recipe(tmp_path)
    target = tmp_path / 'dataset.csv'
    materialize_xml_collection(recipe, target, confirm=True)
    target.write_text(target.read_text(encoding='utf-8').replace('Alpha', 'changed'), encoding='utf-8')
    with pytest.raises(ValueError, match='checksum'):
        read_null_aware_collection_table(target)
    with pytest.raises(ValueError, match='checksum'):
        read_table_page(target)
    with pytest.raises(ValueError, match='checksum'):
        read_dataset_frame(target)


def test_encoding_does_not_reinterpret_normal_csv(tmp_path: Path):
    assert encode_value(None) == NULL_TOKEN
    assert encode_value('') == EMPTY_TOKEN
    assert decode_value(encode_value('\\N')) == '\\N'
    assert decode_value(encode_value('\\E')) == '\\E'
    ordinary = tmp_path / 'generic.csv'
    ordinary.write_text('value\n\\N\n\\E\n', encoding='utf-8')
    assert read_table_page(ordinary)['rows'] == [[r'\N'], [r'\E']]
    with pytest.raises(ValueError, match='regular local files'):
        read_null_aware_collection_table(ordinary)


def test_preview_comments_are_explanations_without_output_files(tmp_path: Path):
    recipe = _recipe(tmp_path)
    result = preview_xml_collection_export(recipe)
    assert result['columns'][-1] == 'Comments'
    assert result['null_encoding']['schema'] == 'ml-lab.xml-null-delimited@1'
    assert result['rows'][0]['value'] is None
    assert json.loads(result['rows'][0]['Comments']) == {'value': 'explicit_xsi_nil'}
    assert result['rows'][1]['value'] == ''
    assert not list(tmp_path.glob('*.csv'))
