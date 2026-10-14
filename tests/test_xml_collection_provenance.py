"""C6 authoritative multi-file events and non-destructive replay contracts."""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from ml_lab.data.xml_collection import inventory_xml_collection
from ml_lab.data.xml_collection_recipe import build_xml_collection_recipe
from ml_lab.data.xml_similarity import compare_xml_collection
from ml_lab.data.xml_collection_export import materialize_xml_collection, XmlCollectionExportError
from ml_lab.data.xml_collection_provenance import (
    XmlCollectionProvenanceError, checkpoint_path, provenance_path,
    verify_collection_event, inspect_collection_recovery,
    regenerate_xml_collection, resume_xml_collection_export,
    start_checkpoint, main,
)


def _recipe(root: Path, *, child: bool = False) -> dict:
    (root / 'a.xml').write_text('<rows><item id="1"><name>Alpha</name><tag>x</tag><tag>y</tag></item></rows>', encoding='utf-8')
    (root / 'b.xml').write_text('<rows><item id="2"><name>Beta</name></item></rows>', encoding='utf-8')
    inventory = inventory_xml_collection(root)
    return build_xml_collection_recipe(
        inventory, compare_xml_collection(inventory), ['a.xml', 'b.xml'], '/rows/item',
        ['text:/rows/item/name', 'text:/rows/item/tag'], allow_missing_fields=True,
        acknowledge_variants=True,
        rules=[{'branch_canonical_path': '/rows/item/tag',
                'strategy': 'separate_table' if child else 'join',
                'separate_table_name': 'tags' if child else '', 'join_delimiter': '|'}],
    )


def test_authoritative_event_has_both_parents_and_child_artifacts(tmp_path: Path):
    recipe = _recipe(tmp_path, child=True)
    output = tmp_path / 'all.csv'
    manifest = materialize_xml_collection(recipe, output, confirm=True)
    assert manifest['provenance']['status'] == 'recorded'
    assert provenance_path(output).is_file()
    assert not checkpoint_path(output).exists()
    event = json.loads(provenance_path(output).read_text(encoding='utf-8'))
    assert event['relation'] == 'derived_from_many' and event['authoritative']
    assert event['source_membership'] == ['a.xml', 'b.xml']
    assert len(event['sources']) == 2 and len(event['outputs']) == 2
    assert event['recipe']['snapshot']['selected_field_ids'] == recipe['selected_field_ids']
    assert verify_collection_event(provenance_path(output))['status'] == 'verified'
    assert manifest['provenance']['sha256']


def test_changed_source_or_output_is_not_accepted(tmp_path: Path):
    recipe = _recipe(tmp_path)
    output = tmp_path / 'all.csv'
    materialize_xml_collection(recipe, output, confirm=True)
    (tmp_path / 'a.xml').write_text('<rows><item><name>different</name></item></rows>', encoding='utf-8')
    assert verify_collection_event(provenance_path(output))['status'] == 'mismatch'
    with pytest.raises(XmlCollectionProvenanceError, match='changed source'):
        regenerate_xml_collection(provenance_path(output), tmp_path / 'redo.csv', confirm=True)
    assert not (tmp_path / 'redo.csv').exists()
    (tmp_path / 'a.xml').write_text('<rows><item id="1"><name>Alpha</name><tag>x</tag><tag>y</tag></item></rows>', encoding='utf-8')
    output.write_text('tampered', encoding='utf-8')
    assert verify_collection_event(provenance_path(output))['status'] == 'mismatch'


def test_regeneration_matches_original_csv_hash(tmp_path: Path):
    recipe = _recipe(tmp_path)
    output = tmp_path / 'original.csv'
    materialize_xml_collection(recipe, output, confirm=True)
    regenerated = regenerate_xml_collection(provenance_path(output), tmp_path / 'reproduced.csv', confirm=True)
    assert regenerated['status'] == 'verified_reproduction'
    assert verify_collection_event(provenance_path(tmp_path / 'reproduced.csv'))['status'] == 'verified'


def test_leftover_checkpoint_replays_if_no_partial_files(tmp_path: Path):
    recipe = _recipe(tmp_path)
    output = tmp_path / 'replay.csv'
    start_checkpoint(recipe, output)
    assert inspect_collection_recovery(output)['status'] == 'ready_to_replay'
    with pytest.raises(XmlCollectionProvenanceError, match='confirmation'):
        resume_xml_collection_export(output)
    manifest = resume_xml_collection_export(output, confirm=True)
    assert manifest['provenance']['status'] == 'recorded'
    assert not checkpoint_path(output).exists()


def test_partial_published_csv_never_trusted_by_recovery(tmp_path: Path):
    recipe = _recipe(tmp_path)
    output = tmp_path / 'partial.csv'
    start_checkpoint(recipe, output)
    output.write_text('partial data', encoding='utf-8')
    assert inspect_collection_recovery(output)['status'] == 'blocked_existing_artifacts'
    with pytest.raises(XmlCollectionProvenanceError, match='blocked'):
        resume_xml_collection_export(output, confirm=True)
    assert output.read_text(encoding='utf-8') == 'partial data'


def test_mismatched_checkpoint_recipe_denied(tmp_path: Path):
    recipe = _recipe(tmp_path)
    output = tmp_path / 'unsafe.csv'
    cp = start_checkpoint(recipe, output)
    payload = json.loads(cp.read_text(encoding='utf-8'))
    payload['recipe']['selected_field_ids'] = []
    cp.write_text(json.dumps(payload), encoding='utf-8')
    with pytest.raises(XmlCollectionProvenanceError, match='recipe hash'):
        inspect_collection_recovery(output)


def test_old_c5_manifests_not_accepted_as_provenance(tmp_path: Path):
    old = tmp_path / 'old.collection.json'
    old.write_text(json.dumps({'schema': 'ml-lab.xml-collection-materialization@1',
                               'provenance': {'status': 'not_recorded'}}), encoding='utf-8')
    with pytest.raises(XmlCollectionProvenanceError, match='authoritative'):
        verify_collection_event(old)


def test_c6_module_cli_verify_and_recovery(tmp_path: Path, capsys):
    recipe = _recipe(tmp_path)
    output = tmp_path / 'cli.csv'
    materialize_xml_collection(recipe, output, confirm=True)
    assert main(['verify', '--provenance', str(provenance_path(output))]) == 0
    assert 'verified' in capsys.readouterr().out
    assert main(['inspect-recovery', '--output', str(output)]) == 0
    assert 'no_checkpoint' in capsys.readouterr().out


def test_publication_failure_rolls_back_and_checkpoint_can_replay(tmp_path: Path, monkeypatch):
    from ml_lab.data import xml_collection_export as exporter
    recipe = _recipe(tmp_path)
    output = tmp_path / 'interrupted.csv'
    actual_replace = exporter.os.replace
    def fail_manifest_publish(src, dest):
        if str(dest).endswith('interrupted.csv.collection.json'):
            raise OSError('simulated failure before final manifest publication')
        return actual_replace(src, dest)
    with monkeypatch.context() as patch:
        patch.setattr(exporter.os, 'replace', fail_manifest_publish)
        with pytest.raises(OSError, match='simulated failure'):
            materialize_xml_collection(recipe, output, confirm=True)
    assert not output.exists()
    assert not provenance_path(output).exists()
    assert checkpoint_path(output).exists()
    assert inspect_collection_recovery(output)['status'] == 'ready_to_replay'
    result = resume_xml_collection_export(output, confirm=True)
    assert result['provenance']['status'] == 'recorded'
    assert verify_collection_event(provenance_path(output))['status'] == 'verified'


def test_existing_target_does_not_create_checkpoint(tmp_path: Path):
    recipe = _recipe(tmp_path)
    output = tmp_path / 'existing.csv'
    output.write_text('do not replace', encoding='utf-8')
    with pytest.raises(FileExistsError):
        materialize_xml_collection(recipe, output, confirm=True)
    assert output.read_text(encoding='utf-8') == 'do not replace'
    assert not checkpoint_path(output).exists()
