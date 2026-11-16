"""ML-UW-24: cross-stage XML acceptance; all sources and outputs stay in tmp_path."""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from ml_lab.data.xml_collection import inventory_xml_collection
from ml_lab.data.xml_collection_review import review_xml_collection
from ml_lab.data.xml_collection_recipe import build_xml_collection_recipe
from ml_lab.data.xml_collection_export import (
    XmlCollectionExportError, materialize_xml_collection, preview_xml_collection_export,
)
from ml_lab.data.xml_collection_scale import materialize_xml_collection_scaled
from ml_lab.data.xml_collection_provenance import provenance_path, verify_collection_event
from ml_lab.data.xml_null_export import read_null_aware_collection_table
from ml_lab.data.xml_similarity import compare_xml_collection
from ml_lab.experiments import ExperimentStore, create_workbench_instance, select_workbench_dataset
from ml_lab.experiments.datasets import describe_selection
from ml_lab.data.types import XmlPreviewColumn
from ml_lab.data.xml_collection_columns import register_collection_columns
from collections import OrderedDict
from ml_lab.data.provenance import read_dataset_frame


NS = '/{urn:science}records/{urn:science}row'
SELECTED = [f'text:{NS}/{{urn:science}}name', f'text:{NS}/{{urn:science}}value',
            f'attribute:{NS}@idx']
ROOT = Path(__file__).resolve().parents[1]
VIEWER_JS = ROOT / 'src/ml_lab/ui/static/ml_lab_xml_group_viewer.js'


def _fixture(folder: Path):
    folder.mkdir(parents=True)
    # Different prefixes refer to the SAME namespace. The two source files
    # deliberately have different display names for the same canonical field.
    (folder / 'a.xml').write_text(
        '<d:records xmlns:d="urn:science" '
        'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">'
        '<d:row idx="1"><d:name>Alpha</d:name><d:value>17</d:value></d:row>'
        '<d:row idx="2"><d:name>Beta</d:name><d:value xsi:nil="true"/></d:row>'
        '</d:records>', encoding='utf-8')
    (folder / 'b.xml').write_text(
        '<q:records xmlns:q="urn:science">'
        '<q:row><q:name>Gamma</q:name></q:row>'
        '<q:row><q:name>Delta</q:name><q:value/></q:row>'
        '</q:records>', encoding='utf-8')
    # Deliberately incompatible expanded root; a valid consolidated union
    # may show it, but an extraction recipe cannot silently merge it.
    (folder / 'other.xml').write_text(
        '<other><entry><name>not-in-recipe</name></entry></other>', encoding='utf-8')
    return folder


def _stages(folder: Path):
    inventory = inventory_xml_collection(folder)
    similarity = compare_xml_collection(inventory)
    record = similarity.to_record()
    union = record['union_structure']
    assert len(union['root_partitions']) == 2
    assert union['source_count'] == 3
    groups = [set(g['members']) for g in union['groups']]
    assert {'a.xml', 'b.xml'} in groups and {'other.xml'} in groups
    assert not union['merge_authorized']
    assert review_xml_collection(similarity, ['a.xml', 'b.xml'])['status'] == 'blocked'
    review = review_xml_collection(similarity, ['a.xml', 'b.xml'], acknowledge_variants=True)
    assert review['status'] == 'ready_for_recipe_design' and not review['merge_authorized']
    recipe = build_xml_collection_recipe(
        inventory, similarity, ['a.xml', 'b.xml'], NS, SELECTED,
        acknowledge_variants=True,
    )
    assert recipe['status'] == 'ready_for_collection_preview'
    assert recipe['allow_missing_fields']
    return inventory, similarity, union, recipe


@pytest.mark.parametrize('extension', ['csv', 'tsv'])
@pytest.mark.parametrize('checkpointed', [False, True])
def test_full_xml_collection_workflow_preserves_nulls_and_lineage(
        tmp_path: Path, extension: str, checkpointed: bool):
    inputs = _fixture(tmp_path / 'input')
    original = {p.name: p.read_bytes() for p in inputs.glob('*.xml')}
    _, _, union, recipe = _stages(inputs)
    missing = next(p for p in union['paths']
                   if p['canonical_path'].endswith('/{urn:science}value'))
    assert missing['present_in'] == ['a.xml', 'b.xml']
    assert missing['occurrences_by_source'] == {'a.xml': 2, 'b.xml': 1}

    preview = preview_xml_collection_export(recipe)
    assert preview['no_files_written']
    assert len(preview['rows']) == 4
    assert preview['columns'] == ['__source_file', '__source_row',
                                  'd:name', 'd:value', '@idx', 'Comments']
    assert json.loads(preview['rows'][1]['Comments']) == {'d:value': 'explicit_xsi_nil'}
    assert json.loads(preview['rows'][2]['Comments']) == {
        'd:value': 'path_absent', '@idx': 'attribute_absent'}
    assert preview['rows'][3]['d:value'] == ''

    target = tmp_path / 'outputs' / f'combined.{extension}'
    target.parent.mkdir()
    export = materialize_xml_collection_scaled if checkpointed else materialize_xml_collection
    manifest = export(recipe, target, confirm=True)
    assert manifest['main_dataset']['row_count'] == 4
    assert manifest['main_dataset']['columns'] == preview['columns']
    assert manifest['main_dataset']['null_encoding']['schema'] == 'ml-lab.xml-null-delimited@1'
    assert verify_collection_event(provenance_path(target))['status'] == 'verified'
    rows = read_null_aware_collection_table(target)
    assert rows[0]['d:name'] == 'Alpha' and rows[0]['d:value'] == '17'
    assert rows[1]['d:name'] == 'Beta' and rows[1]['d:value'] is None
    assert rows[2]['d:name'] == 'Gamma' and rows[2]['d:value'] is None
    assert rows[3]['d:name'] == 'Delta' and rows[3]['d:value'] == ''
    assert json.loads(rows[2]['Comments']) == {
        '@idx': 'attribute_absent', 'd:value': 'path_absent'}
    frame = read_dataset_frame(target)
    assert frame.loc[1, 'd:value'] is None and frame.loc[3, 'd:value'] == ''

    store = ExperimentStore(tmp_path)
    instance = create_workbench_instance(store)
    selected = select_workbench_dataset(
        store, instance['instance_id'], target,
        expected_generation=instance['generation'],
    )
    assert selected['active_dataset']['rows'] == 4
    assert 'Comments' not in describe_selection(store, target)['feature_columns']
    assert {p.name: p.read_bytes() for p in inputs.glob('*.xml')} == original


def test_incompatible_roots_and_changed_sources_fail_closed(tmp_path: Path):
    inputs = _fixture(tmp_path / 'inputs')
    inventory, comparison, _, recipe = _stages(inputs)
    blocked = review_xml_collection(
        comparison, ['a.xml', 'other.xml'], acknowledge_variants=True)
    assert blocked['status'] == 'blocked'
    assert blocked['summary']['incompatible_pair_count'] == 1

    (inputs / 'b.xml').write_text('<different/>', encoding='utf-8')
    with pytest.raises((XmlCollectionExportError, ValueError), match='changed|fingerprint|valid'):
        preview_xml_collection_export(recipe)
    out = tmp_path / 'not-published.csv'
    with pytest.raises((XmlCollectionExportError, ValueError), match='changed|fingerprint|valid'):
        materialize_xml_collection_scaled(recipe, out, confirm=True)
    assert not out.exists()


def test_group_and_union_views_share_observed_canonical_paths(tmp_path: Path):
    if shutil.which('node') is None:
        pytest.skip('Node.js not installed')
    _, _, union, _ = _stages(_fixture(tmp_path / 'input'))
    script = r'''
      const assert = require('node:assert/strict');
      const fs = require('node:fs');
      const {deriveGroupView, deriveConsolidatedView, fieldGroupCoverage} = require(process.argv[1]);
      const union = JSON.parse(fs.readFileSync(0, 'utf8'));
      const all = deriveConsolidatedView(union);
      const group = union.groups.find(g => g.members.length === 2);
      const individual = deriveGroupView(union, group.group_id);
      assert.ok(all && individual);
      assert.equal(all.sourceCount, 3);
      assert.equal(all.children.get(null).length, 2);
      assert.equal(individual.sourceCount, 2);
      assert.equal(individual.nodes.some(n => n.id.includes('/other')), false);
      const name = all.nodes.find(n => n.id.endsWith('/{urn:science}name'));
      assert.ok(name);
      assert.ok(individual.nodeMap.get(name.id));
      assert.deepEqual(fieldGroupCoverage(all, name).map(x => x.sourceCount).sort(), [1, 2]);
      process.stdout.write('OK');
    '''
    proc = subprocess.run(['node', '-e', script, str(VIEWER_JS)],
                          input=json.dumps(union), text=True, capture_output=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout == 'OK'


def test_null_encoding_manifest_detects_tampering(tmp_path: Path):
    _, _, _, recipe = _stages(_fixture(tmp_path / 'input'))
    target = tmp_path / 'combined.csv'
    materialize_xml_collection(recipe, target, confirm=True)
    target.write_text(target.read_text().replace('Gamma', 'corrupt'), encoding='utf-8')
    with pytest.raises(ValueError, match='checksum'):
        read_null_aware_collection_table(target)
    assert verify_collection_event(provenance_path(target))['status'] == 'mismatch'


def test_optional_flask_routes_cover_group_and_union_views(tmp_path: Path):
    flask = pytest.importorskip('flask')
    from ml_lab.ui import create_ui_blueprint
    _fixture(tmp_path / 'input')
    app = flask.Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(create_ui_blueprint(name='mluw24', experiment_project_root=tmp_path),
                           url_prefix='/pah/ml-lab')
    client = app.test_client()
    idle = client.get('/pah/ml-lab/data/xml/collection')
    assert idle.status_code == 200 and b'data-xml-union-tree' not in idle.data
    result = client.post('/pah/ml-lab/data/xml/collection', data={
        'directory': str(tmp_path / 'input'), 'recursive': '1',
        'max_files': '10', 'compare': '1',
    })
    assert result.status_code == 200
    assert b'data-xml-group-viewer' in result.data
    assert b'data-xml-union-viewer' in result.data
    assert b'Graphical structure' in result.data


@pytest.mark.parametrize('checkpointed', [False, True])
def test_repeated_child_rows_align_equivalent_namespaced_columns(tmp_path: Path, checkpointed: bool):
    inputs = tmp_path / 'xml'
    inputs.mkdir()
    for prefix, name, first, second in [('a', 'Alpha', 'x', 'y'),
                                        ('b', 'Beta', 'z', 'w')]:
        (inputs / f'{prefix}.xml').write_text(
            f'<{prefix}:records xmlns:{prefix}="urn:science">'
            f'<{prefix}:row><{prefix}:name>{name}</{prefix}:name>'
            f'<{prefix}:tag>{first}</{prefix}:tag>'
            f'<{prefix}:tag>{second}</{prefix}:tag>'
            f'</{prefix}:row></{prefix}:records>', encoding='utf-8')
    inv = inventory_xml_collection(inputs)
    rep = compare_xml_collection(inv)
    branch = NS + '/{urn:science}tag'
    recipe = build_xml_collection_recipe(
        inv, rep, ['a.xml', 'b.xml'], NS,
        [f'text:{NS}/{{urn:science}}name', 'text:' + branch],
        acknowledge_variants=True,
        rules=[{'branch_canonical_path': branch, 'strategy': 'separate_table',
                'separate_table_name': 'tags'}],
    )
    assert recipe['status'] == 'ready_for_collection_preview'
    preview = preview_xml_collection_export(recipe)
    assert preview['child_tables']['tags']['columns'] == [
        '__source_file', '__parent_record', 'a:tag', 'Comments']
    exporter = materialize_xml_collection_scaled if checkpointed else materialize_xml_collection
    target = tmp_path / 'group.csv'
    result = exporter(recipe, target, confirm=True)
    assert len(result['child_datasets']) == 1
    child = Path(result['child_datasets'][0]['path'])
    values = read_null_aware_collection_table(
        child, manifest=target.with_suffix('.csv.collection.json'))
    assert [row['a:tag'] for row in values] == ['x', 'y', 'z', 'w']
    assert [row['__source_file'] for row in values] == ['a.xml', 'a.xml', 'b.xml', 'b.xml']
    assert verify_collection_event(provenance_path(target))['status'] == 'verified'


def test_canonical_alignment_does_not_combine_different_field_ids():
    headers = OrderedDict()
    seen = {}
    first = XmlPreviewColumn(name='p:value', source_field_ids=('text:/a/value',),
                             source_element_canonical_paths=('/a/value',))
    other = XmlPreviewColumn(name='p:value', source_field_ids=('text:/b/value',),
                             source_element_canonical_paths=('/b/value',))
    assert register_collection_columns([first], headers, seen) == {'p:value': 'p:value'}
    with pytest.raises(ValueError, match='conflicting meanings'):
        register_collection_columns([other], headers, seen)
    reserved = XmlPreviewColumn(name='Comments', source_field_ids=('text:/c',))
    with pytest.raises(ValueError, match='reserved provenance'):
        register_collection_columns([reserved], headers, seen)


@pytest.mark.parametrize('extension', ['csv', 'tsv'])
def test_c5_c7_have_identical_null_aware_table_bytes(tmp_path: Path, extension: str):
    _, _, _, recipe = _stages(_fixture(tmp_path / 'input'))
    regular = tmp_path / f'c5.{extension}'
    scaled = tmp_path / f'c7.{extension}'
    materialize_xml_collection(recipe, regular, confirm=True)
    materialize_xml_collection_scaled(recipe, scaled, confirm=True)
    # The outputs are deterministic even though each export receives its own
    # immutable provenance event and timestamps.
    assert regular.read_bytes() == scaled.read_bytes()
    assert read_null_aware_collection_table(regular) == read_null_aware_collection_table(scaled)
