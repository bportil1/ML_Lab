"""ML-UW-26: pyPIQUE-style UI and artifact-derived workbench state."""
from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from ml_lab.experiments import (
    ExperimentStore, create_workbench_instance,
    get_workbench_instance, select_experiment_dataset, select_workbench_dataset,
)
from ml_lab.experiments.workbench_preflight import inspect_existing_work

UI = Path(__file__).resolve().parents[1] / 'src/ml_lab/ui'


def test_visual_contract_and_shared_grids():
    template = (UI / 'templates/ml_lab_ui/base.html').read_text()
    css = (UI / 'static/ml_lab_pypique_visual.css').read_text()
    tables = (UI / 'static/ml_lab_tables.js').read_text()
    script = (UI / 'static/ml_lab_pypique_preflight.js').read_text()
    assert 'ml_lab_pypique_visual.css' in template
    assert 'ml_lab_pypique_preflight.js' in template
    assert 'data-mlwb-preflight-url' in template
    assert 'workbench_preflight_api' in template
    assert '#102d4f' in css and '#e3ebf2' in css and '#eaf1f6' in css
    assert '.mlwb-page table.mlwb-grid' in css
    assert "document.querySelectorAll('table')" in tables
    assert "table.dataset.mlwbGridReady" in tables
    assert "table.classList.contains('source-table')" in tables
    assert 'Export visible CSV' in tables
    assert "No active dataset — checking saved work" in script
    assert 'nothing is recomputed automatically' not in script  # no false success claims
    assert 'fetch(panel.dataset.mlwbPreflightUrl' in script
    env = Environment(loader=FileSystemLoader(UI / 'templates'))
    env.get_template('ml_lab_ui/base.html')


def test_preflight_reuses_verified_dataset_without_new_run_or_selection(tmp_path: Path):
    source = tmp_path / 'analysis/data/derived/collected.csv'
    source.parent.mkdir(parents=True)
    source.write_text('feature,label\n5,A\n6,B\n', encoding='utf-8')
    store = ExperimentStore(tmp_path)
    previous = select_experiment_dataset(store, source)
    instance = create_workbench_instance(store)
    initial = inspect_existing_work(store, instance)
    assert initial['active_status'] == 'empty'
    assert initial['saved'][0]['status'] == 'ready'
    assert initial['saved'][0]['path'] == 'analysis/data/derived/collected.csv'
    assert initial['steps'][0]['state'] == 'ready'
    assert not initial['derived']  # already described by saved revision
    assert not store.list_runs(previous['experiment_id'])
    assert get_workbench_instance(store, instance['instance_id'])['generation'] == 0

    chosen = select_workbench_dataset(store, instance['instance_id'], source,
                                      expected_generation=0)
    after = inspect_existing_work(store, chosen)
    assert after['active_status'] == 'ready'
    assert after['active']['rows'] == 2
    assert after['steps'][0]['state'] == 'complete'
    assert all(s['state'] == 'ready' for s in after['steps'][1:])
    assert not store.list_runs(previous['experiment_id'])
    assert get_workbench_instance(store, instance['instance_id'])['generation'] == 1


def test_stale_artifact_is_not_suggested_as_reusable(tmp_path: Path):
    source = tmp_path / 'input.csv'
    source.write_text('feature\n3\n4\n', encoding='utf-8')
    store = ExperimentStore(tmp_path)
    select_experiment_dataset(store, source)
    context = create_workbench_instance(store)
    source.write_text('feature\n3\n4\n5\n', encoding='utf-8')
    report = inspect_existing_work(store, context)
    assert report['saved'][0]['status'] == 'stale'
    assert report['saved'][0]['reason']
    assert report['steps'][0]['state'] == 'blocked'


def test_derived_discovery_is_bounded_does_not_follow_symlinks(tmp_path: Path):
    root = tmp_path / 'project'
    root.mkdir()
    derived = root / 'analysis/data/derived'
    derived.mkdir(parents=True)
    for i in range(12):
        (derived / f'output_{i:02d}.csv').write_text('a\n1\n', encoding='utf-8')
    (derived / 'notes.txt').write_text('Do not offer as a table')
    outside = tmp_path / 'outside.csv'
    outside.write_text('secret\n3\n')
    (derived / 'linked.csv').symlink_to(outside)
    store = ExperimentStore(root)
    context = create_workbench_instance(store)
    report = inspect_existing_work(store, context, derived_limit=4)
    assert report['steps'][0]['state'] == 'ready'
    assert len(report['derived']) == 4
    assert all(item['status'] == 'found' for item in report['derived'])
    assert not any('linked' in item['path'] or 'notes' in item['path'] for item in report['derived'])
    assert not store.list_experiments()


def test_preflight_route_is_read_only_and_mount_aware(tmp_path: Path):
    flask = __import__('pytest').importorskip('flask')
    from ml_lab.ui import create_ui_blueprint
    app = flask.Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(create_ui_blueprint(name='pypique_visual_test',
            experiment_project_root=str(tmp_path)), url_prefix='/tools/ml-lab')
    client = app.test_client()
    result = client.get('/tools/ml-lab/api/workbench/preflight')
    assert result.status_code == 200
    assert result.json['schema'] == 'ml-lab.artifact-preflight@1'
    before = client.get('/tools/ml-lab/api/workbench/context').json
    client.get('/tools/ml-lab/api/workbench/preflight')
    after = client.get('/tools/ml-lab/api/workbench/context').json
    assert before['generation'] == after['generation']
    assert before['status'] == after['status']
    html = client.get('/tools/ml-lab/data').get_data(as_text=True)
    assert '/tools/ml-lab/api/workbench/preflight' in html
    assert 'ml_lab_pypique_visual.css' in html


def test_large_saved_dataset_is_discovered_without_automatic_full_read(tmp_path: Path, monkeypatch):
    source = tmp_path / 'very_large.csv'
    source.write_bytes(b'feature\n1\n' * 500_000)  # 5 MB
    store = ExperimentStore(tmp_path)
    saved = select_experiment_dataset(store, source)
    context = create_workbench_instance(store)
    import ml_lab.experiments.workbench_preflight as preflight
    def must_not_recompute(*args, **kwargs):
        raise AssertionError('Opening a page must not re-profile/reparse large saved data')
    monkeypatch.setattr(preflight, 'verify_selection', must_not_recompute)
    report = inspect_existing_work(store, context)
    assert report['saved'][0]['experiment_id'] == saved['experiment_id']
    assert report['saved'][0]['status'] == 'found'
    assert report['saved'][0]['reason'] == 'Large saved file; verify on selection'
    assert report['steps'][0]['state'] == 'ready'
    assert store.list_runs(saved['experiment_id']) == []


def test_existing_feature_profile_is_recognized_from_cache_without_recomputation(tmp_path: Path):
    from ml_lab.experiments.feature_profile_cache import feature_baseline_key, save_feature_report
    source = tmp_path / 'input.csv'
    source.write_text('a,b\n1,2\n3,4\n', encoding='utf-8')
    store = ExperimentStore(tmp_path)
    instance = create_workbench_instance(store)
    selected = select_workbench_dataset(store, instance['instance_id'], source,
                                        expected_generation=0)
    key = feature_baseline_key(selected, overrides={})
    assert save_feature_report(store, selected, key, {
        'schema':'ml-lab.workbench-feature-explorer@1',
        'relationships':[], 'columns':[], 'summary':{}
    }, preserve_cache_key=key)
    result = inspect_existing_work(store, selected)
    assert result['profile_cached'] is True
    assert next(s for s in result['steps'] if s['id']=='profiling')['state']=='complete'
