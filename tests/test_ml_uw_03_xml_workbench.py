"""ML-UW-03: cross-mode XML workflow, provenance and workbench handoff."""
from __future__ import annotations

from pathlib import Path

import pytest

from ml_lab.data.xml_collection import inventory_xml_collection
from ml_lab.data.xml_collection_recipe import build_xml_collection_recipe
from ml_lab.data.xml_similarity import compare_xml_collection
from ml_lab.data.xml_collection_export import materialize_xml_collection
from ml_lab.experiments import (
    DatasetSelectionError, ExperimentStore, create_workbench_instance,
    get_workbench_instance, select_workbench_dataset,
)


def _materialized(tmp_path: Path) -> tuple[Path, Path]:
    (tmp_path / "one.xml").write_text(
        '<rows><item><value>10</value></item></rows>', encoding='utf-8'
    )
    (tmp_path / "two.xml").write_text(
        '<rows><item><value>20</value></item></rows>', encoding='utf-8'
    )
    inventory = inventory_xml_collection(tmp_path)
    recipe = build_xml_collection_recipe(
        inventory, compare_xml_collection(inventory),
        ['one.xml', 'two.xml'], '/rows/item',
        ['text:/rows/item/value'],
    )
    assert recipe['status'] == 'ready_for_collection_preview'
    output = tmp_path / 'analysis' / 'data' / 'combined.csv'
    materialize_xml_collection(recipe, output, confirm=True)
    return output, tmp_path / 'one.xml'


def test_xml_c6_export_can_be_pinned_with_verified_lineage(tmp_path):
    output, first_xml = _materialized(tmp_path)
    store = ExperimentStore(tmp_path)
    instance = create_workbench_instance(store)
    selected = select_workbench_dataset(
        store, instance['instance_id'], output, expected_generation=0,
    )
    assert selected['status'] == 'ready'
    assert selected['active_dataset']['path'] == 'analysis/data/combined.csv'
    assert selected['active_dataset']['rows'] == 2
    assert selected['active_dataset']['provenance']['event_id'].startswith('event:sha256:')
    assert get_workbench_instance(store, instance['instance_id'])['status'] == 'ready'

    first_xml.write_text('<rows><item><value>99</value></item></rows>', encoding='utf-8')
    stale = get_workbench_instance(store, instance['instance_id'])
    assert stale['status'] == 'stale'
    assert stale['active_dataset'] is None
    assert 'changed' in stale['validation_error'].lower()


def test_xml_c6_export_rejects_tampered_output_and_sidecar(tmp_path):
    output, _ = _materialized(tmp_path)
    store = ExperimentStore(tmp_path)
    instance = create_workbench_instance(store)
    output.write_text('value\n999\n', encoding='utf-8')
    with pytest.raises(DatasetSelectionError, match='changed'):
        select_workbench_dataset(store, instance['instance_id'], output,
                                 expected_generation=0)
    assert store.list_experiments() == []


def test_xml_c6_collection_export_rejects_other_project(tmp_path):
    output, _ = _materialized(tmp_path)
    elsewhere = tmp_path / 'another_project'
    elsewhere.mkdir()
    store = ExperimentStore(elsewhere)
    instance = create_workbench_instance(store)
    with pytest.raises(DatasetSelectionError, match='project-local'):
        select_workbench_dataset(store, instance['instance_id'], output,
                                 expected_generation=0)


def test_xml_workbench_and_manual_activation_routes(tmp_path):
    flask = pytest.importorskip('flask')
    from ml_lab.ui import create_ui_blueprint

    project = tmp_path / 'project'
    project.mkdir()
    output, first_xml = _materialized(project)
    app = flask.Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(
        create_ui_blueprint(name='research', experiment_project_root=project),
        url_prefix='/ml',
    )
    client = app.test_client()
    landing = client.get('/ml/data/xml/workbench')
    assert landing.status_code == 200
    assert b'Single XML file' in landing.data and b'XML collection' in landing.data
    opened = client.post('/ml/data/xml/open', data={'source': str(first_xml)})
    assert opened.status_code == 302
    assert '/data/xml/' in opened.headers['Location']
    assert client.post('/ml/data/xml/open', data={'source': '/nonexistent.xml'}).status_code == 302
    form = client.get('/ml/workbench/dataset/select-file', query_string={'path': str(output)})
    assert form.status_code == 200
    assert b'Activate selected dataset' in form.data
    context = client.get('/ml/api/workbench/context').get_json()
    from ml_lab.experiments import describe_selection
    selection = describe_selection(ExperimentStore(project), output)
    accepted = client.post('/ml/workbench/dataset/select-file', data={
        'path': str(output),
        'source_sha256': selection['sha256'],
        'source_dataset_id': selection['dataset_id'],
        'expected_generation': str(context['generation']),
        'features': selection['column_names'],
    })
    assert accepted.status_code == 200
    assert b'Active dataset updated' in accepted.data
    assert client.get('/ml/api/workbench/context').get_json()['status'] == 'ready'
    rejected = client.post('/ml/workbench/dataset/select-file', data={
        'path': str(output),
        'source_sha256': selection['sha256'],
        'source_dataset_id': selection['dataset_id'],
        'expected_generation': str(context['generation']),
        'features': selection['column_names'],
    })
    assert rejected.status_code == 409
