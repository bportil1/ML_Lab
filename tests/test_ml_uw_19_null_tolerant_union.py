"""ML-UW-19: optional/missing XML evidence produces an observed union, not fragmentation."""
from __future__ import annotations

import json
from pathlib import Path

from ml_lab.data import (
    catalog_xml_null_fields,
    compare_xml_collection,
    inventory_xml_collection,
    preview_xml_null_provenance,
)
from ml_lab.data.xml_collection_review import review_xml_collection
from ml_lab.data.xml_ingest import inspect_xml_structure
from ml_lab.data.xml_similarity import UNION_SCHEMA, save_xml_collection_comparison


def _collection(folder: Path, left: str, right: str):
    (folder / 'a.xml').write_text(left, encoding='utf-8')
    (folder / 'b.xml').write_text(right, encoding='utf-8')
    return compare_xml_collection(inventory_xml_collection(folder)).to_record()


def _rows(items: str) -> str:
    return '<rows>' + items + '</rows>'


def _items(fields: str, n: int = 2) -> str:
    return ''.join('<item id="' + str(i) + '">' + fields + '</item>' for i in range(n))


def _union_path(union, path):
    return next(entry for entry in union['paths'] if entry['canonical_path'] == path)


def test_low_path_overlap_null_tolerant_and_complete_link_group(tmp_path):
    left_fields = '<common>stable</common>' + ''.join(f'<only_left_{i}>{i}</only_left_{i}>' for i in range(8))
    right_fields = '<common>stable</common>' + ''.join(f'<only_right_{i}>{i}</only_right_{i}>' for i in range(8))
    record = _collection(tmp_path, _rows(_items(left_fields)), _rows(_items(right_fields)))
    pair = record['pairs'][0]
    assert pair['similarity_score'] < 0.80
    assert pair['below_similarity_threshold'] is True
    assert pair['classification'] == 'compatible_variant_candidate'
    assert pair['reason'] == 'null_tolerant_shared_record_root'
    assert pair['differences']['left_only_path_count'] == 8
    assert pair['differences']['right_only_path_count'] == 8
    assert record['summary']['group_count'] == 1
    assert record['groups'][0]['members'] == ['a.xml', 'b.xml']
    assert review_xml_collection(record, ['a.xml', 'b.xml'])['status'] == 'blocked'
    reviewed = review_xml_collection(record, ['a.xml', 'b.xml'], acknowledge_variants=True)
    assert reviewed['status'] == 'ready_for_recipe_design'
    assert not reviewed['merge_authorized'] and not reviewed['extraction_recipe_validated']

    union = record['union_structure']
    assert union['schema'] == UNION_SCHEMA and union['source_count'] == 2
    assert union['path_count'] == 19
    assert union['merge_authorized'] is False
    only_a = _union_path(union, '/rows/item/only_left_0')
    assert only_a['present_in'] == ['a.xml']
    assert only_a['source_count'] == 1 and only_a['missing_source_count'] == 1
    assert only_a['total_element_occurrences'] == 2
    assert only_a['occurrences_by_source'] == {'a.xml': 2}
    assert only_a['parent_canonical_path'] == '/rows/item'
    group = union['groups'][0]
    assert group['members'] == ['a.xml', 'b.xml']
    assert set(group['path_ids']) == {x['canonical_path'] for x in union['paths']}
    assert '/rows/item/only_left_0' in group['optional_path_ids']
    assert '/rows/item/common' not in group['optional_path_ids']
    assert not group['merge_authorized']

    # Both union field IDs and the ML-UW-18 null catalog use canonical paths.
    artifacts = {name: inspect_xml_structure(tmp_path / name) for name in ('a.xml', 'b.xml')}
    catalog = catalog_xml_null_fields(artifacts, '/rows/item')
    preview = preview_xml_null_provenance(tmp_path / 'b.xml', artifacts['b.xml'], catalog, 'b.xml')
    absent = next(cell for cell in preview['records'][0]['cells']
                  if cell['field_id'] == 'text:/rows/item/only_left_0')
    assert absent['status'] == 'path_absent'
    assert absent['observations'][0]['value'] is None
    assert absent['observations'][0]['null_reason'] == 'path_absent'


def test_missing_value_nil_and_optionality_do_not_fragment(tmp_path):
    a = _rows(_items('<value>10</value><name>ok</name>'))
    b = _rows('''<item id="0"><value xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
                            xsi:nil="true"/><name>ok</name></item>
                  <item id="1"><name>ok</name></item>''')
    record = _collection(tmp_path, a, b)
    pair = record['pairs'][0]
    assert pair['classification'] == 'compatible_variant_candidate'
    assert pair['differences']['text_presence_change_count'] >= 1
    assert pair['differences']['optional_presence_change_count'] >= 1
    assert pair['differences']['cardinality_change_count'] == 0
    assert record['summary']['group_count'] == 1
    union = record['union_structure']
    value = _union_path(union, '/rows/item/value')
    assert value['occurrences_by_source'] == {'a.xml': 2, 'b.xml': 1}
    assert value['text_occurrences_by_source'] == {'a.xml': 2, 'b.xml': 0}
    assert value['total_element_occurrences'] == 3
    nil_attr = next(a for a in value['attributes'] if a['attribute_qname'].endswith('}nil'))
    assert nil_attr['present_in'] == ['b.xml']
    assert nil_attr['total_occurrences'] == 1


def test_root_mixed_content_and_repeated_shape_remain_diagnostic(tmp_path):
    roots = _collection(tmp_path, _rows(_items('<v>1</v>')), '<other><record/></other>')
    assert roots['pairs'][0]['classification'] == 'incompatible'
    assert len(roots['union_structure']['root_partitions']) == 2
    assert roots['summary']['group_count'] == 2

    a = _rows(_items('<v>1</v>'))
    b = _rows(_items('<v>1</v>tail'))
    mixed = _collection(tmp_path, a, b)
    assert mixed['pairs'][0]['classification'] == 'review_required'
    assert mixed['pairs'][0]['reason'] == 'mixed_content_shape_differs'
    assert mixed['summary']['group_count'] == 2

    a = _rows(_items('<v>1</v><v>2</v>'))
    b = _rows(_items('<v>1</v>'))
    repeated = _collection(tmp_path, a, b)
    assert repeated['pairs'][0]['classification'] == 'review_required'
    assert repeated['pairs'][0]['reason'] == 'observed_cardinality_differs'
    assert repeated['summary']['group_count'] == 2


def test_single_record_variants_group_for_review_without_auto_approval(tmp_path):
    # A source with one observed record may not pass the record-candidate
    # heuristic; missing fields should not force artificial singleton groups.
    record = _collection(tmp_path,
                         '<records><row><id>1</id><name>A</name></row></records>',
                         '<records><row><id>2</id><age>32</age></row></records>')
    assert record['pairs'][0]['reason'] == 'no_shared_record_root_evidence'
    assert record['pairs'][0]['classification'] == 'review_required'
    assert record['summary']['group_count'] == 1
    assert record['groups'][0]['classification'] == 'review_required'
    assert record['groups'][0]['shared_record_root_candidates'] == []
    assert record['groups'][0]['merge_authorized'] is False
    assert review_xml_collection(record, ['a.xml', 'b.xml'])['status'] == 'blocked'
    reviewed = review_xml_collection(record, ['a.xml', 'b.xml'], acknowledge_variants=True)
    assert reviewed['status'] == 'ready_for_recipe_design'
    assert reviewed['merge_authorized'] is False
    union_group = record['union_structure']['groups'][0]
    assert '/records/row/id' in union_group['path_ids']
    assert '/records/row/name' in union_group['optional_path_ids']
    assert '/records/row/age' in union_group['optional_path_ids']


def test_namespace_counts_exclusions_and_saved_union_are_stable(tmp_path):
    a = '<x:rows xmlns:x="urn:demo"><x:item code="a"/><x:item code="b"/></x:rows>'
    b = '<y:rows xmlns:y="urn:demo"><y:item code="z" extra="1"/></y:rows>'
    report = _collection(tmp_path, a, b)
    union = report['union_structure']
    item = _union_path(union, '/{urn:demo}rows/{urn:demo}item')
    assert item['total_element_occurrences'] == 3
    assert item['occurrences_by_source'] == {'a.xml': 2, 'b.xml': 1}
    assert item['parent_canonical_path'] == '/{urn:demo}rows'
    code = next(attr for attr in item['attributes'] if attr['attribute_qname'] == 'code')
    assert code['total_occurrences'] == 3
    extra = next(attr for attr in item['attributes'] if attr['attribute_qname'] == 'extra')
    assert extra['missing_source_count'] == 1
    assert extra['total_occurrences'] == 1
    assert set(union['root_partitions'][0]['members']) == {'a.xml', 'b.xml'}
    assert json.loads(json.dumps(union)) == union

    path = save_xml_collection_comparison(compare_xml_collection(inventory_xml_collection(tmp_path)),
                                          tmp_path / 'comparison.json')
    written = json.loads(path.read_text(encoding='utf-8'))
    assert written['union_structure'] == union
    assert not written['summary']['merge_authorized']

    inventory = inventory_xml_collection(tmp_path)
    (tmp_path / 'b.xml').write_text('<changed/>')
    stale = compare_xml_collection(inventory).to_record()
    assert stale['summary']['excluded_source_count'] == 1
    assert stale['union_structure']['source_count'] == 1
    assert stale['union_structure']['source_paths'] == ['a.xml']
    assert not any('b.xml' in path['present_in'] for path in stale['union_structure']['paths'])


def test_ui_contract_documents_advisory_threshold():
    page = (Path(__file__).parents[1] / 'src/ml_lab/ui/templates/ml_lab_ui/xml_collection_inventory.html')
    text = page.read_text(encoding='utf-8')
    assert 'Similarity reference (0–1)' in text
    assert 'missing paths, absent attributes and empty values' in text
    assert 'Inspect canonical union structure' in text
