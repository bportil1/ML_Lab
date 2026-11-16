"""ML-UW-22: C2 group-view visualization consumes observed union evidence only."""
from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess

import pytest

from ml_lab.data import compare_xml_collection, inventory_xml_collection


JS = Path(__file__).resolve().parents[1] / 'src/ml_lab/ui/static/ml_lab_xml_group_viewer.js'
TEMPLATE = Path(__file__).resolve().parents[1] / 'src/ml_lab/ui/templates/ml_lab_ui/xml_collection_inventory.html'


def _comparison(folder: Path):
    (folder / 'a.xml').write_text('''<r:rows xmlns:r="https://example.test/ns">
      <r:item code="one"><r:id>1</r:id><r:optional>yes</r:optional></r:item>
      <r:item code="two"><r:id>2</r:id><r:optional>yes</r:optional></r:item>
    </r:rows>''', encoding='utf-8')
    (folder / 'b.xml').write_text('''<p:rows xmlns:p="https://example.test/ns">
      <p:item><p:id>3</p:id></p:item><p:item><p:id>4</p:id></p:item>
    </p:rows>''', encoding='utf-8')
    (folder / 'c.xml').write_text('<other><entry>unrelated</entry></other>', encoding='utf-8')
    return compare_xml_collection(inventory_xml_collection(folder)).to_record()


def test_graph_data_group_isolation_optional_fields_namespace_and_attributes(tmp_path):
    if shutil.which('node') is None:
        pytest.skip('Node.js is optional for local JS derivation tests')
    union = _comparison(tmp_path)['union_structure']
    javascript = r'''
      const assert = require('node:assert/strict');
      const fs = require('node:fs');
      const {deriveGroupView, decodedSegment, namespaceFor} = require(process.argv[1]);
      const union = JSON.parse(fs.readFileSync(0, 'utf8'));
      const paired = union.groups.find(g => g.members.length === 2);
      const other = union.groups.find(g => g.members.length === 1);
      assert.ok(paired && other);
      const view = deriveGroupView(union, paired.group_id);
      assert.equal(view.sourceCount, 2);
      assert.equal(view.nodes.some(n => n.id.includes('/other')), false);
      const opt = view.nodes.find(n => n.label === 'optional');
      assert.ok(opt);
      assert.deepEqual(opt.present, ['a.xml']);
      assert.equal(opt.missing, 1);
      assert.equal(opt.occurrences, 2);
      assert.equal(opt.namespace, 'https://example.test/ns');
      assert.equal(view.optionalElementCount >= 1, true);
      assert.equal(view.nodes.some(n => n.kind === 'element' && n.recordRootCandidate), true);
      const attr = view.nodes.find(n => n.label === '@code');
      assert.ok(attr);
      assert.equal(attr.kind, 'attribute');
      assert.equal(attr.missing, 1);
      assert.deepEqual(attr.present, ['a.xml']);
      assert.equal(attr.occurrences, 2);
      assert.equal(attr.parent.endsWith('}item'.replaceAll('/', '~1')), true);
      assert.equal(view.children.get(attr.parent).some(n => n.id === attr.id), true);
      const single = deriveGroupView(union, other.group_id);
      assert.equal(single.sourceCount, 1);
      assert.equal(single.nodes.some(n => n.id === opt.id), false);
      assert.equal(deriveGroupView(union, 'missing-group'), null);
      assert.equal(decodedSegment('/{https:~1~1example.test~1ns}item'), '{https://example.test/ns}item');
      assert.equal(namespaceFor('{https://example.test/ns}item'), 'https://example.test/ns');
      process.stdout.write('group viewer derivation: OK');
    '''
    run = subprocess.run(['node', '-e', javascript, str(JS)], input=json.dumps(union),
                         text=True, capture_output=True, check=False)
    assert run.returncode == 0, run.stderr
    assert 'OK' in run.stdout


def test_template_has_read_only_selection_and_one_canonical_union_payload():
    source = TEMPLATE.read_text(encoding='utf-8')
    assert 'data-xml-open-group' in source
    assert 'data-xml-group-select' in source
    assert 'data-xml-group-search' in source
    assert 'data-xml-group-tree' in source
    assert 'data-xml-group-inspector' in source
    assert 'data-xml-group-union' in source
    assert source.count('comparison.union_structure|tojson(indent=2)') == 1
    assert "filename='ml_lab_xml_group_viewer.js'" in source
    javascript = JS.read_text(encoding='utf-8')
    assert 'fetch(' not in javascript
    assert 'innerHTML' not in javascript
    assert 'textContent' in javascript


def test_flask_comparison_page_has_group_structure_controls(tmp_path):
    pytest.importorskip('flask')
    from ml_lab.ui import create_app
    _comparison(tmp_path)
    client = create_app(config={'TESTING': True}).test_client()
    response = client.post('/data/xml/collection', data={
        'directory': str(tmp_path), 'compare': '1', 'recursive': '1',
    })
    assert response.status_code == 200
    assert b'Graphical structure' in response.data
    assert b'data-xml-group-viewer' in response.data
    assert b'ml_lab_xml_group_viewer.js' in response.data
    assert b'data-xml-open-group' in response.data
