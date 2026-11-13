"""ML-UW-18 canonical path and explicit null provenance acceptance tests."""
from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from ml_lab.data.xml_ingest import XmlSafetyError, inspect_xml_structure
from ml_lab.data.xml_null_provenance import (
    INVENTORY_SCHEMA, NULL_REASONS, PREVIEW_SCHEMA,
    XmlNullProvenanceError, catalog_xml_null_fields, preview_xml_null_provenance,
)


def _sources(tmp_path: Path):
    a = tmp_path / 'a.xml'
    b = tmp_path / 'b.xml'
    a.write_text('''<doc xmlns:a="urn:shared" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
    <record code="R1"><a:name>Alpha</a:name><quantity>10</quantity><empty></empty>
       <status xsi:nil="true"/><many>one</many><many>two</many></record>
    <record><a:name>Beta</a:name><quantity/><status>ok</status></record>
    </doc>''', encoding='utf-8')
    b.write_text('''<doc xmlns:b="urn:shared"><record><b:name>Gamma</b:name>
      <extra key="E1">Extra</extra></record></doc>''', encoding='utf-8')
    aa, bb = inspect_xml_structure(a), inspect_xml_structure(b)
    root = '/doc/record'
    inventory = catalog_xml_null_fields({'a.xml': aa, 'b.xml': bb}, root)
    return a, b, aa, bb, inventory


def _cell(record, suffix):
    return next(cell for cell in record['cells'] if cell['field_id'].endswith(suffix))


def test_canonical_union_and_source_presence(tmp_path):
    a, b, aa, bb, inventory = _sources(tmp_path)
    assert inventory['schema'] == INVENTORY_SCHEMA
    assert inventory['source_count'] == 2
    assert all(field['column_key'] == field['field_id'] for field in inventory['fields'])
    name = next(field for field in inventory['fields'] if field['field_id'] == 'text:/doc/record/{urn:shared}name')
    assert name['present_in'] == ['a.xml', 'b.xml']
    assert name['missing_in'] == []
    quantity = next(field for field in inventory['fields'] if field['field_id'] == 'text:/doc/record/quantity')
    assert quantity['present_in'] == ['a.xml']
    assert quantity['missing_in'] == ['b.xml']
    assert json.loads(json.dumps(inventory))['fields'] == inventory['fields']


def test_per_record_conditions_and_lineage(tmp_path):
    a, b, aa, bb, inventory = _sources(tmp_path)
    a_data = preview_xml_null_provenance(a, aa, inventory, 'a.xml')
    b_data = preview_xml_null_provenance(b, bb, inventory, 'b.xml')
    assert a_data['schema'] == PREVIEW_SCHEMA
    assert a_data['registered_dataset'] is False and not a_data['source_modified']
    assert [r['record_number'] for r in a_data['records']] == [1, 2]
    first, second = a_data['records']
    assert _cell(first, 'quantity')['observations'][0]['value'] == '10'
    nil = _cell(first, 'status')
    assert nil['status'] == 'explicit_xsi_nil'
    assert nil['observations'][0]['value'] is None
    assert nil['observations'][0]['null_reason'] in NULL_REASONS
    assert _cell(first, 'empty')['status'] == 'empty_element'
    assert _cell(first, 'empty')['observations'][0]['value'] == ''
    assert _cell(second, 'quantity')['status'] == 'empty_element'
    assert _cell(second, '@code')['status'] == 'attribute_absent'
    assert _cell(second, '@code')['observations'][0]['value'] is None
    assert _cell(second, 'empty')['status'] == 'path_absent'
    repeated = _cell(first, 'many')
    assert repeated['status'] == 'multiple' and repeated['occurrence_count'] == 2
    assert [o['value'] for o in repeated['observations']] == ['one', 'two']
    assert [o['occurrence_address'] for o in repeated['observations']] == [[0, 0], [0, 1]]
    only_b = b_data['records'][0]
    assert _cell(only_b, 'quantity')['status'] == 'path_absent'
    assert _cell(only_b, '@code')['status'] == 'attribute_absent'
    assert _cell(first, '@key')['status'] == 'path_absent'
    assert _cell(only_b, 'name')['observations'][0]['value'] == 'Gamma'


def test_preview_limit_and_non_mutation(tmp_path):
    a, b, aa, bb, inventory = _sources(tmp_path)
    original = a.read_bytes()
    result = preview_xml_null_provenance(a, aa, inventory, 'a.xml', max_records=1)
    assert result['truncated'] is True and result['preview_record_count'] == 1
    assert a.read_bytes() == original
    for invalid in (0, 501, True, -1):
        with pytest.raises(XmlNullProvenanceError, match='max_records'):
            preview_xml_null_provenance(a, aa, inventory, 'a.xml', max_records=invalid)


def test_stale_source_bad_inventory_and_escape_rejected(tmp_path):
    a, b, aa, bb, inventory = _sources(tmp_path)
    with pytest.raises(XmlNullProvenanceError, match='source is not part'):
        preview_xml_null_provenance(a, aa, inventory, 'wrong.xml')
    tampered = copy.deepcopy(inventory)
    tampered['fields'][0]['source_element_canonical_path'] = '/outside'
    with pytest.raises(XmlNullProvenanceError, match='outside'):
        preview_xml_null_provenance(a, aa, tampered, 'a.xml')
    a.write_text('<doc><record><quantity>changed</quantity></record></doc>', encoding='utf-8')
    with pytest.raises(XmlNullProvenanceError, match='changed'):
        preview_xml_null_provenance(a, aa, inventory, 'a.xml')


def test_parse_errors_not_null_records(tmp_path):
    a, b, aa, bb, inventory = _sources(tmp_path)
    broken = tmp_path / 'broken.xml'
    broken.write_text('<doc><record>', encoding='utf-8')
    with pytest.raises(Exception):
        inspect_xml_structure(broken)
    malicious = tmp_path / 'malicious.xml'
    malicious.write_text('<!DOCTYPE doc [<!ENTITY x "v">]><doc><record>&x;</record></doc>', encoding='utf-8')
    with pytest.raises(XmlSafetyError):
        inspect_xml_structure(malicious)


def test_namespaces_do_not_collapse_different_uris(tmp_path):
    p = tmp_path / 'namespaced.xml'
    p.write_text('<doc xmlns:a="urn:a" xmlns:b="urn:b"><record><a:x>A</a:x><b:x>B</b:x></record></doc>')
    art = inspect_xml_structure(p)
    inv = catalog_xml_null_fields({'source': art}, '/doc/record')
    keys = [f['field_id'] for f in inv['fields']]
    assert 'text:/doc/record/{urn:a}x' in keys
    assert 'text:/doc/record/{urn:b}x' in keys
    rows = preview_xml_null_provenance(p, art, inv, 'source')['records']
    assert {f['observations'][0]['value'] for f in rows[0]['cells']} == {'A', 'B'}
