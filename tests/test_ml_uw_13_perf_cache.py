"""ML-UW-13: reuse read-only scientific results, preserve data and metrics."""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from ml_lab.data.feature_explorer import explore_features, explore_cached_views, _pair_preview
from ml_lab.experiments import ExperimentStore, create_workbench_instance, select_workbench_dataset
from ml_lab.experiments.feature_profile_cache import (
    feature_baseline_key, feature_report_key, load_feature_report, save_feature_report,
)


def _prepared(tmp_path):
    file = tmp_path / 'sample.csv'
    rows = ['sample_id,x,y,label,flag']
    for i in range(1, 151):
        rows.append(f'item-{i},{i},{i % 11},{"ABC"[i % 3]},{"true" if i % 2 else "false"}')
    file.write_text('\n'.join(rows) + '\n', encoding='utf-8')
    store = ExperimentStore(tmp_path)
    initial = create_workbench_instance(store)
    context = select_workbench_dataset(store, initial['instance_id'], file, expected_generation=0)
    return file, store, context


def test_cached_selection_views_identical_to_fresh_profile_and_do_not_rerun_relationships(tmp_path, monkeypatch):
    file, store, context = _prepared(tmp_path)
    sha = context['active_dataset']['sha256']
    source_bytes = file.read_bytes()
    initial = explore_features(str(file), source_sha256=sha,
                               selected='x', left='x', right='y')
    baseline_key = feature_baseline_key(context, overrides={})
    view_key = feature_report_key(context, selected='x', left='x', right='y', overrides={})
    assert baseline_key != view_key
    assert save_feature_report(store, context, baseline_key, initial)
    assert load_feature_report(store, context, baseline_key) == initial
    variants = [('y', 'x', 'label'), ('label', 'x', 'y'), ('flag', 'label', 'flag')]
    expected = [explore_features(str(file), source_sha256=sha, selected=a, left=b, right=c)
                for a, b, c in variants]
    import ml_lab.data.feature_explorer as explorer

    def fail_profile(*args, **kwargs):
        raise AssertionError('cached preview must not recompute A2 profile/MI')
    monkeypatch.setattr(explorer, 'profile_file', fail_profile)
    for (a, b, c), reference in zip(variants, expected):
        replay = explore_cached_views(str(file), source_sha256=sha,
                                      baseline=load_feature_report(store, context, baseline_key),
                                      selected=a, left=b, right=c)
        assert replay == reference
        assert save_feature_report(store, context, feature_report_key(
            context, selected=a, left=b, right=c, overrides={}), replay)
    assert file.read_bytes() == source_bytes
    with sqlite3.connect(store.db_path) as db:
        assert db.execute('PRAGMA user_version').fetchone()[0] == 3
        assert db.execute('SELECT COUNT(*) FROM workbench_feature_profile_cache').fetchone()[0] == 4


def test_cached_views_validate_source_and_scoping(tmp_path):
    source, store, context = _prepared(tmp_path)
    original = explore_features(str(source), source_sha256=context['active_dataset']['sha256'])
    assert feature_baseline_key(context, overrides={}) != feature_baseline_key(
        {**context, 'generation': context['generation'] + 1}, overrides={})
    assert feature_baseline_key(context, overrides={}) != feature_baseline_key(
        context, overrides={'x': 'discrete'})
    with pytest.raises(ValueError, match='Invalid cached profile'):
        explore_cached_views(str(source), source_sha256=context['active_dataset']['sha256'],
                             baseline={'schema': 'missing'}, selected='x', left='x', right='y')
    source.write_text(source.read_text() + 'last,999,1,A,true\n', encoding='utf-8')
    with pytest.raises(ValueError, match='changed'):
        explore_cached_views(str(source), source_sha256=context['active_dataset']['sha256'],
                             baseline=original, selected='x', left='x', right='y')


def test_sparse_contingency_keeps_existing_counts_without_dense_crosstab(monkeypatch):
    # 5000 unique row and column labels would create a 25 million-cell
    # crosstab, despite the workbench showing no more than 8 x 8 cells.
    frame = pd.DataFrame({
        'left': [f'left-{i}' for i in range(5000)],
        'right': [f'right-{i}' for i in range(5000)],
    })
    expected = _pair_preview(frame.iloc[:30], 'left', 'right', {'left': 'discrete', 'right': 'discrete'})
    assert expected['max_cell'] == 1

    def disallow_dense(*args, **kwargs):
        raise AssertionError('dense crosstab must not be allocated')
    monkeypatch.setattr(pd, 'crosstab', disallow_dense)
    output = _pair_preview(frame, 'left', 'right', {'left': 'discrete', 'right': 'discrete'})
    assert output['kind'] == 'contingency'
    assert output['max_cell'] == 1
    assert output['truncated'] is True
    assert len(output['rows']) == 8
    assert all(len(row['cells']) == 8 for row in output['rows'])


def test_sparse_contingency_matches_reference_including_offscreen_maximum():
    frame = pd.DataFrame({
        'left': ['a'] * 12 + ['b'] * 8 + ['c'] * 10 + ['d'] * 12,
        'right': ['x'] * 4 + ['y'] * 4 + ['z'] * 4 + ['x'] * 8 + ['q'] * 10 + ['w'] * 12,
    })
    expected = pd.crosstab(frame['left'], frame['right'])
    output = _pair_preview(frame, 'left', 'right', {'left': 'discrete', 'right': 'discrete'})
    assert output['max_cell'] == int(expected.max().max())
    for row in output['rows']:
        for category, value in zip(output['y_categories'], row['cells']):
            assert value == int(expected.loc[row['label'], category])


def test_sqlite_cache_bounds_keep_baseline_after_many_selection_changes(tmp_path):
    source, store, context = _prepared(tmp_path)
    baseline = explore_features(str(source), source_sha256=context['active_dataset']['sha256'])
    key = feature_baseline_key(context, overrides={})
    assert save_feature_report(store, context, key, baseline, preserve_cache_key=key)
    for i in range(35):
        variant = feature_report_key(context, selected=f'display-{i}', left='x', right='y', overrides={})
        assert save_feature_report(store, context, variant, baseline, preserve_cache_key=key)
    assert load_feature_report(store, context, key) == baseline
    with sqlite3.connect(store.db_path) as db:
        assert db.execute('SELECT COUNT(*) FROM workbench_feature_profile_cache').fetchone()[0] == 24
        assert db.execute('PRAGMA user_version').fetchone()[0] == 3
