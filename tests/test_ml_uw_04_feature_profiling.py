"""ML-UW-04: active-dataset profiling, stored display roles, and pair geometry."""
from __future__ import annotations

import sqlite3

import pytest

from ml_lab.data import profile_file
from ml_lab.data.feature_explorer import explore_features
from ml_lab.experiments import (
    ExperimentStore, create_workbench_instance, get_workbench_instance,
    select_workbench_dataset,
)
from ml_lab.experiments.feature_roles import feature_roles, set_feature_role
from ml_lab.experiments.workbench import WorkbenchConflict, WorkbenchInstanceError


def _dataset(project, name='selected.csv'):
    file = project / name
    file.write_text(
        'id,x,y,group,flag,zero\n'
        'one,0,1,A,true,0\n'
        'two,1,3,A,false,0\n'
        'three,2,5,A,true,0\n'
        'four,3,7,B,false,0\n'
        'five,4,9,B,true,1\n'
        'six,5,11,B,false,0\n'
        'seven,6,13,C,true,0\n'
        'eight,7,15,C,false,0\n', encoding='utf-8',
    )
    return file


def _selected(tmp_path):
    source = _dataset(tmp_path)
    store = ExperimentStore(tmp_path)
    instance = create_workbench_instance(store)
    active = select_workbench_dataset(store, instance['instance_id'], source, expected_generation=0)
    return source, store, active


def test_additive_variance_zero_rate_and_existing_profile_contract(tmp_path):
    source = _dataset(tmp_path)
    record = profile_file(source).to_record()
    assert record['schema'] == 'ml-lab.data-profile@1'
    columns = {column['name']: column for column in record['columns']}
    assert columns['x']['numeric_summary']['variance'] == pytest.approx(6.0)
    assert columns['x']['numeric_summary']['zero_rate'] == pytest.approx(1 / 8)
    assert columns['group']['numeric_summary'] == {}


def test_roles_are_persistent_but_do_not_change_source_or_profile(tmp_path):
    source, store, active = _selected(tmp_path)
    before = source.read_bytes()
    context_id = active['instance_id']
    selected = active['active_dataset']
    overrides = set_feature_role(store, context_id, column='x', role='discrete',
                                 source_sha256=selected['sha256'], expected_generation=1)
    assert overrides == {'x': 'discrete'}
    reopened = ExperimentStore(tmp_path)
    context = get_workbench_instance(reopened, context_id)
    assert feature_roles(reopened, context) == {'x': 'discrete'}
    assert context['generation'] == 1
    report = explore_features(str(source), source_sha256=selected['sha256'], overrides=overrides,
                              selected='x', left='x', right='group')
    assert report['distribution']['kind'] == 'categories'
    assert report['pair']['kind'] == 'contingency'
    assert next(col for col in report['columns'] if col['name'] == 'x')['inferred_type'] == 'integer'
    assert source.read_bytes() == before
    set_feature_role(reopened, context_id, column='x', role='auto',
                     source_sha256=selected['sha256'], expected_generation=1)
    assert feature_roles(reopened, context) == {}


def test_type_aware_pair_explorer_and_invalid_roles(tmp_path):
    source, store, active = _selected(tmp_path)
    sha = active['active_dataset']['sha256']
    report = explore_features(str(source), source_sha256=sha, selected='y', left='x', right='y')
    assert report['distribution']['kind'] == 'histogram'
    boolean_col = next(col for col in report['columns'] if col['name'] == 'zero')
    assert boolean_col['zero_rate'] == pytest.approx(7 / 8)
    assert report['pair']['kind'] == 'scatter' and len(report['pair']['points']) == 8
    pair = explore_features(str(source), source_sha256=sha, left='group', right='y')
    assert pair['pair']['kind'] == 'groups'
    assert [x['label'] for x in pair['pair']['groups']] == ['A', 'B', 'C']
    with pytest.raises(WorkbenchInstanceError, match='numeric'):
        set_feature_role(store, active['instance_id'], column='group', role='continuous',
                         source_sha256=sha, expected_generation=1)
    with pytest.raises(WorkbenchInstanceError, match='Unknown'):
        set_feature_role(store, active['instance_id'], column='missing', role='discrete',
                         source_sha256=sha, expected_generation=1)
    with pytest.raises(WorkbenchConflict, match='changed'):
        set_feature_role(store, active['instance_id'], column='x', role='discrete',
                         source_sha256=sha, expected_generation=0)


def test_role_metadata_scopes_to_sha_and_rejects_stale_source(tmp_path):
    source, store, active = _selected(tmp_path)
    sha = active['active_dataset']['sha256']
    set_feature_role(store, active['instance_id'], column='x', role='discrete',
                     source_sha256=sha, expected_generation=1)
    source.write_text(source.read_text() + 'nine,8,17,C,false,0\n', encoding='utf-8')
    stale = get_workbench_instance(store, active['instance_id'])
    assert stale['status'] == 'stale' and feature_roles(store, stale) == {}
    with pytest.raises(WorkbenchInstanceError, match='verified'):
        set_feature_role(store, active['instance_id'], column='x', role='continuous',
                         source_sha256=sha, expected_generation=1)
    with pytest.raises(ValueError, match='changed'):
        explore_features(str(source), source_sha256=sha)
    new = select_workbench_dataset(store, active['instance_id'], source, expected_generation=1)
    assert new['status'] == 'ready'
    assert feature_roles(store, new) == {}
    with sqlite3.connect(store.db_path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 3


def test_optional_role_table_remains_v3_compatible_on_rollback(tmp_path):
    store = ExperimentStore(tmp_path)
    with sqlite3.connect(store.db_path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 3
        db.execute('DROP TABLE workbench_feature_roles')
    reopened = ExperimentStore(tmp_path)
    with sqlite3.connect(reopened.db_path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 3
        assert db.execute("SELECT name FROM sqlite_master WHERE name='workbench_feature_roles'").fetchone()


def test_feature_route_static_template_contract(tmp_path):
    from jinja2 import Environment, FileSystemLoader
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / 'src' / 'ml_lab' / 'ui' / 'templates'
    Environment(loader=FileSystemLoader(path)).get_template('ml_lab_ui/workbench_feature_profiling.html')


def test_browser_mounted_blueprint_and_role_post(tmp_path):
    flask = pytest.importorskip('flask')
    from ml_lab.ui import create_ui_blueprint

    source = _dataset(tmp_path)
    app = flask.Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(create_ui_blueprint(name='mount', experiment_project_root=tmp_path), url_prefix='/ml')
    client = app.test_client()
    empty = client.get('/ml/workbench/feature-profiling')
    assert empty.status_code == 200 and b'No verified active dataset' in empty.data
    context = client.get('/ml/api/workbench/context').get_json()
    store = ExperimentStore(tmp_path)
    select_workbench_dataset(store, context['instance_id'], source, expected_generation=0)
    page = client.get('/ml/workbench/feature-profiling?inspect=1')
    assert page.status_code == 200
    assert b'Per-feature metadata' in page.data
    sha = get_workbench_instance(store, context['instance_id'])['active_dataset']['sha256']
    change = client.post('/ml/workbench/feature-profiling/role', data={
        'column': 'x', 'role': 'discrete', 'source_sha256': sha, 'expected_generation': '1',
    })
    assert change.status_code == 302
    assert feature_roles(store, get_workbench_instance(store, context['instance_id'])) == {'x': 'discrete'}
    bad = client.post('/ml/workbench/feature-profiling/role', data={
        'column': 'x', 'role': 'discrete', 'source_sha256': sha, 'expected_generation': '0',
    })
    assert bad.status_code == 409
    assert client.get('/ml/workbench/feature-profiling/json').get_json()['schema'] == 'ml-lab.workbench-feature-explorer@1'
