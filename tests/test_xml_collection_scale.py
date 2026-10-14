"""XML-C7 source-by-source scaling and safe replay tests (no network)."""
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from ml_lab.data.xml_collection import inventory_xml_collection
from ml_lab.data.xml_collection_recipe import build_xml_collection_recipe
from ml_lab.data.xml_similarity import compare_xml_collection
from ml_lab.data.xml_collection_provenance import verify_collection_event, provenance_path
from ml_lab.data.xml_collection_scale import (
    XmlCollectionExportError, describe_sqlite_handoff,
    inspect_xml_collection_scaled, materialize_xml_collection_scaled,
)


def _ready(tmp_path: Path, *, child=False, count=3):
    for idx in range(count):
        tags = '<tag>t0</tag><tag>t1</tag>' if idx % 2 == 0 else ''
        (tmp_path / f's{idx}.xml').write_text(
            f'<records><record><name>N{idx}</name>{tags}</record></records>', encoding='utf-8')
    inventory = inventory_xml_collection(tmp_path)
    return build_xml_collection_recipe(
        inventory, compare_xml_collection(inventory), [f's{i}.xml' for i in range(count)],
        '/records/record', ['text:/records/record/name', 'text:/records/record/tag'],
        allow_missing_fields=True, acknowledge_variants=True,
        rules=[{'branch_canonical_path': '/records/record/tag',
                'strategy': 'separate_table' if child else 'join',
                'join_delimiter': ',', 'separate_table_name': 'tags' if child else ''}],
    )


def test_scaled_export_and_c6_provenance(tmp_path: Path):
    recipe = _ready(tmp_path, child=True)
    output = tmp_path / 'combined.csv'
    events = []
    manifest = materialize_xml_collection_scaled(recipe, output, confirm=True, progress=events.append)
    assert manifest['main_dataset']['row_count'] == 3
    assert verify_collection_event(provenance_path(output))['status'] == 'verified'
    assert [x['phase'] for x in events] == ['started', 'source_completed', 'source_completed',
                                            'source_completed', 'published']
    assert inspect_xml_collection_scaled(output)['status'] == 'no_checkpoint'
    with output.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 3 and [r['name'] for r in rows] == ['N0', 'N1', 'N2']
    assert [r['__source_file'] for r in rows] == ['s0.xml', 's1.xml', 's2.xml']
    child = manifest['child_datasets']
    assert len(child) == 1 and child[0]['row_count'] == 4
    handoff = describe_sqlite_handoff(manifest)
    assert handoff['database_created'] is False and handoff['storage_owner'] == 'ml_lab'
    assert len(handoff['datasets']) == 2


def test_failure_then_resume_skips_verified_source(tmp_path: Path, monkeypatch):
    from ml_lab.data import xml_collection_scale as mod
    recipe = _ready(tmp_path)
    output = tmp_path / 'out.tsv'
    extract = mod._extract_part
    calls = []

    def fail_on_second(path, source, validated, **kwargs):
        calls.append(source['relative_path'])
        if source['relative_path'] == 's1.xml':
            raise RuntimeError('simulated crash')
        return extract(path, source, validated, **kwargs)

    with monkeypatch.context() as p:
        p.setattr(mod, '_extract_part', fail_on_second)
        with pytest.raises(RuntimeError, match='crash'):
            materialize_xml_collection_scaled(recipe, output, confirm=True)
    assert calls == ['s0.xml', 's1.xml']
    assert not output.exists()
    assert inspect_xml_collection_scaled(output)['completed_sources'] == 1
    manifest = materialize_xml_collection_scaled(recipe, output, confirm=True, resume=True)
    assert manifest['main_dataset']['row_count'] == 3
    assert verify_collection_event(provenance_path(output))['status'] == 'verified'
    assert inspect_xml_collection_scaled(output)['status'] == 'no_checkpoint'


def test_tampered_part_rejected_on_resume(tmp_path: Path, monkeypatch):
    from ml_lab.data import xml_collection_scale as mod
    recipe = _ready(tmp_path)
    output = tmp_path / 'out.csv'
    extract = mod._extract_part

    def fail_on_second(path, source, validated, **kwargs):
        if source['relative_path'] == 's1.xml':
            raise RuntimeError('stop')
        return extract(path, source, validated, **kwargs)

    with monkeypatch.context() as p:
        p.setattr(mod, '_extract_part', fail_on_second)
        with pytest.raises(RuntimeError):
            materialize_xml_collection_scaled(recipe, output, confirm=True)
    directory = output.parent / f'.{output.name}.c7.parts'
    (directory / '000000.json').write_text('{}', encoding='utf-8')
    with pytest.raises(XmlCollectionExportError, match='corrupted'):
        materialize_xml_collection_scaled(recipe, output, confirm=True, resume=True)
    assert not output.exists()


def test_changed_source_blocks_replay(tmp_path: Path, monkeypatch):
    from ml_lab.data import xml_collection_scale as mod
    recipe = _ready(tmp_path)
    output = tmp_path / 'out.csv'
    with monkeypatch.context() as p:
        p.setattr(mod, '_extract_part', lambda *a, **kw: (_ for _ in ()).throw(RuntimeError('stop')))
        with pytest.raises(RuntimeError):
            materialize_xml_collection_scaled(recipe, output, confirm=True)
    (tmp_path / 's0.xml').write_text('<records/>', encoding='utf-8')
    with pytest.raises(XmlCollectionExportError, match='changed|validation|no longer valid'):
        materialize_xml_collection_scaled(recipe, output, confirm=True, resume=True)
    assert not output.exists()


def test_limits_confirm_and_existing_output(tmp_path: Path):
    recipe = _ready(tmp_path)
    output = tmp_path / 'out.csv'
    with pytest.raises(XmlCollectionExportError, match='confirmation'):
        materialize_xml_collection_scaled(recipe, output)
    with pytest.raises(XmlCollectionExportError, match='maximum rows'):
        materialize_xml_collection_scaled(recipe, output, confirm=True, max_total_rows=2)
    assert not output.exists()
    assert inspect_xml_collection_scaled(output)['status'] == 'interrupted'
    output.write_text('do not overwrite', encoding='utf-8')
    with pytest.raises(FileExistsError):
        materialize_xml_collection_scaled(recipe, output, confirm=True, resume=True)
    assert output.read_text() == 'do not overwrite'


def test_progress_callback_failure_does_not_change_results(tmp_path: Path):
    recipe = _ready(tmp_path)
    def monitor(_):
        raise RuntimeError('viewer broken')
    output = tmp_path / 'out.csv'
    result = materialize_xml_collection_scaled(recipe, output, confirm=True, progress=monitor)
    assert result['main_dataset']['row_count'] == 3
    assert verify_collection_event(provenance_path(output))['status'] == 'verified'


def test_scaled_regeneration_compares_hashes(tmp_path: Path):
    from ml_lab.data.xml_collection_scale import regenerate_xml_collection_scaled
    recipe = _ready(tmp_path)
    output = tmp_path / 'original.csv'
    materialize_xml_collection_scaled(recipe, output, confirm=True)
    result = regenerate_xml_collection_scaled(provenance_path(output), tmp_path / 'reproduced.csv', confirm=True)
    assert result['status'] == 'verified_reproduction'


def test_two_simultaneous_exports_blocked(tmp_path: Path):
    from ml_lab.data.xml_collection_scale import _output_lock
    output = tmp_path / 'busy.csv'
    recipe = _ready(tmp_path)
    with _output_lock(output):
        with pytest.raises(XmlCollectionExportError, match='already running'):
            materialize_xml_collection_scaled(recipe, output, confirm=True)
