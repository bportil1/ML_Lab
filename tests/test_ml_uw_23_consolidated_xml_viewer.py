"""ML-UW-23: one consolidated graphical union, preserving observed source evidence."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from ml_lab.data import compare_xml_collection, inventory_xml_collection

ROOT = Path(__file__).resolve().parents[1]
JS = ROOT / "src/ml_lab/ui/static/ml_lab_xml_group_viewer.js"
PAGE = ROOT / "src/ml_lab/ui/templates/ml_lab_ui/xml_collection_inventory.html"


def _union(folder: Path):
    # Namespace prefixes differ while expanded QNames remain the same.
    (folder / "a.xml").write_text('''<r:rows xmlns:r="urn:records">
        <r:item flag="1"><r:id>1</r:id><r:optional>A</r:optional></r:item>
        <r:item flag="2"><r:id>2</r:id><r:optional>B</r:optional></r:item>
    </r:rows>''', encoding="utf-8")
    (folder / "b.xml").write_text('''<x:rows xmlns:x="urn:records">
        <x:item><x:id>3</x:id></x:item><x:item><x:id>4</x:id></x:item>
    </x:rows>''', encoding="utf-8")
    (folder / "c.xml").write_text('''<other><entry code="c">other</entry></other>''', encoding="utf-8")
    return compare_xml_collection(inventory_xml_collection(folder)).to_record()["union_structure"]


def _node_run(script: str, union: dict):
    if not shutil.which("node"):
        pytest.skip("Node.js not installed")
    completed = subprocess.run(
        ["node", "-e", script, str(JS)], input=json.dumps(union),
        text=True, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout


def test_union_includes_all_groups_and_preserves_namespace_optional_paths(tmp_path):
    union = _union(tmp_path)
    script = r'''
        const assert = require('node:assert/strict');
        const fs = require('node:fs');
        const {deriveConsolidatedView, deriveGroupView, fieldGroupCoverage, rootOf} = require(process.argv[1]);
        const union = JSON.parse(fs.readFileSync(0, 'utf8'));
        const view = deriveConsolidatedView(union);
        assert.ok(view);
        assert.equal(view.sourceCount, 3);
        assert.equal(view.groupCount, union.groups.length);
        assert.equal(view.nodes.filter(n => n.kind === 'element').length, union.path_count);
        assert.equal(view.children.get(null).length, 2);
        const optional = view.nodes.find(n => n.label === 'optional');
        assert.ok(optional);
        assert.equal(optional.namespace, 'urn:records');
        assert.deepEqual(optional.present, ['a.xml']);
        assert.equal(optional.missing, 2);
        assert.equal(optional.occurrences, 2);
        const group = union.groups.find(g => g.members.includes('a.xml'));
        const groupView = deriveGroupView(union, group.group_id);
        assert.ok(groupView.nodes.find(n => n.id === optional.id));
        const flag = view.nodes.find(n => n.label === '@flag');
        assert.ok(flag);
        assert.equal(flag.parent.endsWith('}item'), true);
        assert.equal(rootOf(flag, view), rootOf(optional, view));
        const cov = fieldGroupCoverage(view, optional);
        assert.equal(cov.length, union.groups.length);
        assert.equal(cov.reduce((sum, c) => sum + c.present, 0), 1);
        assert.equal(cov.reduce((sum, c) => sum + c.missing, 0), 2);
        const pair = cov.find(c => c.groupId === group.group_id);
        assert.ok(pair);
        assert.equal(pair.sourceCount, 2);
        assert.equal(pair.present, 1);
        assert.equal(pair.missing, 1);
        const unrelated = view.nodes.find(n => n.label === 'entry');
        assert.ok(unrelated);
        assert.notEqual(rootOf(unrelated, view), rootOf(optional, view));
        assert.equal(fieldGroupCoverage(groupView, optional).length, 0);
        process.stdout.write('consolidated union: OK');
    '''
    assert "OK" in _node_run(script, union)


def test_union_distinguishes_cross_group_shared_from_group_specific():
    # Directly exercise a minimal, valid union contract with one common root.
    union = {
        'groups': [
            {'group_id': 'G1', 'members': ['a.xml', 'b.xml'], 'shared_record_root_candidates': []},
            {'group_id': 'G2', 'members': ['c.xml'], 'shared_record_root_candidates': []},
        ],
        'source_paths': ['a.xml', 'b.xml', 'c.xml'],
        'paths': [
            {'canonical_path': '/root', 'parent_canonical_path': None,
             'present_in': ['a.xml', 'b.xml', 'c.xml'], 'occurrences_by_source': {'a.xml': 1, 'b.xml': 1, 'c.xml': 1},
             'text_occurrences_by_source': {}, 'optional_in': [], 'repeated_in': [], 'mixed_content_in': [], 'attributes': []},
            {'canonical_path': '/root/leaf', 'parent_canonical_path': '/root',
             'present_in': ['a.xml'], 'occurrences_by_source': {'a.xml': 1},
             'text_occurrences_by_source': {}, 'optional_in': ['a.xml'], 'repeated_in': [], 'mixed_content_in': [], 'attributes': []},
        ],
    }
    script = r'''
        const assert = require('node:assert/strict');
        const fs = require('node:fs');
        const {deriveConsolidatedView, fieldGroupCoverage} = require(process.argv[1]);
        const union = JSON.parse(fs.readFileSync(0, 'utf8'));
        const view = deriveConsolidatedView(union);
        const common = fieldGroupCoverage(view, view.nodeMap.get('/root'));
        assert.deepEqual(common.map(g => [g.present, g.missing]), [[2,0],[1,0]]);
        const specific = fieldGroupCoverage(view, view.nodeMap.get('/root/leaf'));
        assert.deepEqual(specific.map(g => [g.present, g.missing, g.optional]), [[1,1,1],[0,1,0]]);
        // Corrupt membership is rejected rather than falsely labelling source coverage.
        union.groups[1].members = ['b.xml', 'c.xml'];
        assert.equal(deriveConsolidatedView(union), null);
        process.stdout.write('group coverage: OK');
    '''
    assert "OK" in _node_run(script, union)


def test_shared_viewer_and_union_markup_are_read_only():
    source = PAGE.read_text(encoding='utf-8')
    js = JS.read_text(encoding='utf-8')
    for name in ['data-xml-union-viewer', 'data-xml-union-root', 'data-xml-union-tree',
                 'data-xml-union-inspector', 'data-xml-union-search', 'data-xml-union-status']:
        assert name in source
    for name in ['data-xml-group-viewer', 'data-xml-group-select', 'data-xml-group-tree',
                 'data-xml-group-inspector', 'data-xml-group-search']:
        assert name in source
    assert source.count('comparison.union_structure|tojson(indent=2)') == 1
    assert source.count("filename='ml_lab_xml_group_viewer.js'") == 1
    assert 'mountViewer(' in js and 'deriveGroupView(' in js and 'deriveConsolidatedView(' in js
    assert 'fetch(' not in js and 'innerHTML' not in js
    assert 'textContent' in js
    if shutil.which('node'):
        assert subprocess.run(['node', '--check', str(JS)], capture_output=True).returncode == 0


def test_flask_collection_view_displays_both_tree_modes(tmp_path):
    pytest.importorskip('flask')
    from ml_lab.ui import create_app
    _union(tmp_path)
    client = create_app(config={'TESTING': True}).test_client()
    result = client.post('/data/xml/collection', data={
        'directory': str(tmp_path), 'compare': '1', 'recursive': '1',
    })
    assert result.status_code == 200
    assert b'data-xml-group-viewer' in result.data
    assert b'data-xml-union-viewer' in result.data
    assert b'Graphical structure' in result.data
