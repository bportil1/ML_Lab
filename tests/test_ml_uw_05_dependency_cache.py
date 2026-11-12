"""ML-UW-05: reuse A2 dependence results and versioned project-local previews."""
from __future__ import annotations

import sqlite3

import pytest

from ml_lab.data.feature_dependency import dependency_matrix
from ml_lab.data.feature_explorer import explore_features
from ml_lab.experiments import (
    ExperimentStore, create_workbench_instance, get_workbench_instance,
    select_workbench_dataset,
)
from ml_lab.experiments.feature_profile_cache import (
    feature_report_key, load_feature_report, save_feature_report,
)
from ml_lab.experiments.feature_roles import set_feature_role
from ml_lab.experiments.workbench import WorkbenchConflict


def _prepared(tmp_path):
    src = tmp_path / 'observations.csv'
    src.write_text('id,x,y,z,label\n'
                   'one,1,14,5,A\n'
                   'two,2,12,5,A\n'
                   'three,3,10,5,B\n'
                   'four,4,8,5,B\n'
                   'five,5,6,5,C\n'
                   'six,6,4,5,C\n'
                   'seven,7,2,5,C\n', encoding='utf-8')
    store = ExperimentStore(tmp_path)
    fresh = create_workbench_instance(store)
    context = select_workbench_dataset(store, fresh['instance_id'], src, expected_generation=0)
    return src, store, context


def test_dependency_matrix_is_symmetric_read_only_and_not_causal(tmp_path):
    source, store, context = _prepared(tmp_path)
    report = explore_features(str(source), source_sha256=context['active_dataset']['sha256'])
    matrix = dependency_matrix(report['relationships'], report['columns'], metric='pearson', limit=12)
    assert matrix['names']
    assert 'id' not in matrix['names']  # A2 source-identified ID is excluded
    assert 'z' not in matrix['names']  # A2 constant has no estimated Pearson coefficient
    ix, iy = matrix['names'].index('x'), matrix['names'].index('y')
    a, b = matrix['rows'][ix]['cells'][iy], matrix['rows'][iy]['cells'][ix]
    assert a == b | {'other': 'y'} or a['value'] == b['value']
    assert a['value'] == pytest.approx(-1)
    assert a['sign'] == 'negative'
    assert matrix['rows'][ix]['cells'][ix]['value'] is None
    mi = dependency_matrix(report['relationships'], report['columns'], metric='normalized_mutual_information')
    assert mi['rows'][ix]['cells'][iy]['value'] is not None
    assert report['summary']['relationship_count'] == len(report['relationships'])
    with pytest.raises(ValueError, match='metric'):
        dependency_matrix(report['relationships'], report['columns'], metric='causality')
    with pytest.raises(ValueError, match='size'):
        dependency_matrix(report['relationships'], report['columns'], limit=100)


def test_cache_hit_and_metric_changes_use_one_profile(tmp_path):
    source, store, context = _prepared(tmp_path)
    key = feature_report_key(context, selected='x', left='x', right='y', overrides={})
    assert load_feature_report(store, context, key) is None
    report = explore_features(str(source), source_sha256=context['active_dataset']['sha256'],
                              selected='x', left='x', right='y')
    assert save_feature_report(store, context, key, report)
    hit = load_feature_report(ExperimentStore(tmp_path), context, key)
    assert hit['schema'] == report['schema']
    assert hit['relationships'] == report['relationships']
    for metric in ('pearson', 'spearman', 'mutual_information', 'normalized_mutual_information'):
        matrix = dependency_matrix(hit['relationships'], hit['columns'], metric=metric)
        assert matrix['metric'] == metric
    assert feature_report_key(context, selected='y', left='x', right='y', overrides={}) != key
    assert feature_report_key(context, selected='x', left='y', right='x', overrides={}) != key
    assert feature_report_key(context, selected='x', left='x', right='y', overrides={'x': 'discrete'}) != key
    with sqlite3.connect(store.db_path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 3
        assert db.execute('SELECT COUNT(*) FROM workbench_feature_profile_cache').fetchone()[0] == 1


def test_role_change_clears_cache_and_stale_generation_cannot_write(tmp_path):
    source, store, context = _prepared(tmp_path)
    sha = context['active_dataset']['sha256']
    key = feature_report_key(context, selected='x', left='x', right='y', overrides={})
    report = explore_features(str(source), source_sha256=sha)
    save_feature_report(store, context, key, report)
    set_feature_role(store, context['instance_id'], column='x', role='discrete',
                     source_sha256=sha, expected_generation=1)
    assert load_feature_report(store, context, key) is None
    updated = get_workbench_instance(store, context['instance_id'])
    with pytest.raises(WorkbenchConflict, match='changed'):
        save_feature_report(store, {**context, 'generation': 0}, key, report)
    updated_key = feature_report_key(updated, selected='x', left='x', right='y', overrides={'x': 'discrete'})
    save_feature_report(store, updated, updated_key, report)
    assert load_feature_report(store, updated, updated_key) is not None
    # A changed source must fail the verified context gate before route cache access.
    source.write_text(source.read_text() + 'eight,8,0,5,B\n', encoding='utf-8')
    assert get_workbench_instance(store, context['instance_id'])['status'] == 'stale'


def test_cache_bounded_and_optional_v3_rollback(tmp_path):
    src, store, context = _prepared(tmp_path)
    report = explore_features(str(src), source_sha256=context['active_dataset']['sha256'])
    for i in range(29):
        key = feature_report_key(context, selected=f'choice-{i}', left='x', right='y', overrides={})
        assert save_feature_report(store, context, key, report)
    with sqlite3.connect(store.db_path) as db:
        assert db.execute('SELECT COUNT(*) FROM workbench_feature_profile_cache').fetchone()[0] == 24
        assert db.execute('PRAGMA user_version').fetchone()[0] == 3
        db.execute('DROP TABLE workbench_feature_profile_cache')
    reopened = ExperimentStore(tmp_path)
    with sqlite3.connect(reopened.db_path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 3
        assert db.execute("SELECT name FROM sqlite_master WHERE name='workbench_feature_profile_cache'").fetchone()


def test_mounted_feature_route_matrix_and_cache(tmp_path):
    flask = pytest.importorskip('flask')
    from ml_lab.ui import create_ui_blueprint

    src, store, _ = _prepared(tmp_path)
    app = flask.Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(create_ui_blueprint(name='ml05', experiment_project_root=tmp_path), url_prefix='/ml')
    client = app.test_client()
    # The browser gets its own explicit, independent cookie-scoped workbench.
    instance = client.get('/ml/api/workbench/context').get_json()
    select_workbench_dataset(store, instance['instance_id'], src, expected_generation=0)
    first = client.get('/ml/workbench/feature-profiling?inspect=1')
    assert first.status_code == 200
    assert b'Dependency matrix' in first.data and b'Profile computed' in first.data
    second = client.get('/ml/workbench/feature-profiling?inspect=1&metric=pearson&matrix_limit=20')
    assert second.status_code == 200
    assert b'Cached profile' in second.data and b'Pearson' in second.data
    new = client.get('/ml/workbench/feature-profiling?inspect=1&refresh=1')
    assert new.status_code == 200 and b'Profile computed' in new.data
    bad = client.get('/ml/workbench/feature-profiling/json?metric=unknown')
    assert bad.status_code == 409 and b'Unknown dependency metric' in bad.data
