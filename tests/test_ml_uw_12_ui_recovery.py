"""ML-UW-12: read-only stale notices, error recovery and mounted route safety."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest
from jinja2 import Environment, FileSystemLoader, select_autoescape


ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / 'src/ml_lab/ui/templates'
STATIC = ROOT / 'src/ml_lab/ui/static'


def test_ui_templates_compile_and_errors_escape_untrusted_paths():
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)),
                      autoescape=select_autoescape(['html']))
    for template in (TEMPLATES / 'ml_lab_ui').glob('*.html'):
        env.get_template('ml_lab_ui/' + template.name)
    warning = env.get_template('ml_lab_ui/_workbench_error.html').module.show_error(
        '<script>malicious</script>', '/pah/ml-lab/workbench/representation',
        '/pah/ml-lab/data', title='Check dataset',
    )
    assert 'role="alert"' in warning
    assert '<script>' not in warning
    assert '&lt;script&gt;' in warning
    assert 'Reload current view' in warning
    assert 'Unsaved' in warning or 'unsaved' in warning
    for name in ('workbench_dataset_select', 'workbench_experiment_activate',
                 'workbench_feature_profiling', 'workbench_representation',
                 'workbench_autoencoder', 'workbench_comparison',
                 'workbench_general_ml_tools'):
        markup = (TEMPLATES / 'ml_lab_ui' / f'{name}.html').read_text()
        assert "_workbench_error.html" in markup
        assert 'mlwb_error.show_error(' in markup
    base = (TEMPLATES / 'ml_lab_ui/base.html').read_text()
    assert 'href="#mlwb-main"' in base
    assert 'id="mlwb-main" tabindex="-1"' in base
    assert 'data-mlwb-context-url' in base


def test_no_auto_mutation_and_local_recheck_on_pageshow():
    node = shutil.which('node')
    if not node:
        pytest.skip('Local Node.js runtime is unavailable')
    js = STATIC / 'ml_lab_workbench_context.js'
    harness = r'''
        const fs = require('fs');
        const vm = require('vm');
        const assert = require('assert');
        const script = fs.readFileSync(process.argv[1], 'utf8');
        const listeners = {};
        const contexts = [
          {status:'ready', generation:1, experiment_revision:1,
           active_dataset:{path:'/private/example.csv', sha256:'sha1'}},
          {status:'ready', generation:2, experiment_revision:2,
           active_dataset:{path:'/private/other.csv', sha256:'sha2'}},
        ];
        const state = {fetches:0};
        const header = {dataset:{}, textContent:'',
          getAttribute: () => '/pah/ml-lab/api/workbench/context'};
        function form(source, datasetSelection = false) {
          const row = {warning:null, firstChild:null,
            querySelector: key => key.includes('source_sha256') ?
              (source ? {value:source}:null) :
              (key.includes('source_dataset_id') ?
                (datasetSelection ? {value:'dataset1'}:null) : row.warning),
            insertBefore: el => {row.warning=el;},
          };
          return row;
        }
        const representation = form('sha1');
        const selection = form('newSha', true);
        const forms = [{value:'1', form:representation},
                       {value:'1', form:selection}];
        const doc = {visibilityState:'visible',
          querySelector: () => header,
          querySelectorAll: () => forms,
          createElement: tag => ({tag, hidden:false,
            children:[], setAttribute(){},
            appendChild(el) {this.children.push(el);}}),
          addEventListener: (key,fn) => {listeners[key]=fn;},
        };
        const win = {location:{href:'/pah/ml-lab/workbench/representation'},
          addEventListener: (key,fn) => {listeners[key]=fn;},
          fetch: (url, opts) => {
            assert.equal(url, '/pah/ml-lab/api/workbench/context');
            assert.equal(opts.cache, 'no-store');
            return Promise.resolve({ok:true, json:()=>Promise.resolve(
              contexts[Math.min(state.fetches++, contexts.length-1)]
            )});
          },
        };
        vm.runInNewContext(script, {window:win,document:doc});
        const tick = () => new Promise(resolve => setImmediate(resolve));
        (async () => {
          await tick();
          assert.equal(state.fetches, 1);
          assert(header.textContent.includes('example.csv'));
          assert.equal(representation.warning, null);
          assert.equal(selection.warning, null); // new dataset fingerprint is intentional
          listeners.pageshow();
          await tick();
          assert.equal(state.fetches, 2);
          assert(header.textContent.includes('other.csv'));
          assert(representation.warning);
          assert.equal(representation.warning.hidden, false);
          assert(selection.warning);
          assert.equal(selection.warning.hidden, false);
          assert.equal(representation.warning.children[1].href,
            '/pah/ml-lab/workbench/representation');
          assert.equal(state.fetches, 2);
          doc.visibilityState='hidden';
          listeners.visibilitychange();
          await tick();
          assert.equal(state.fetches, 2);
        })().catch(e => { console.error(e); process.exitCode=1; });
    '''
    result = subprocess.run([node, '-e', textwrap.dedent(harness), str(js)],
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stdout + '\n' + result.stderr


def test_workbench_select_file_uses_typed_conflict_status():
    source = (ROOT / 'src/ml_lab/ui/workbench_routes.py').read_text()
    code = source[source.index('    @blueprint.route("/workbench/dataset/select-file"'):]
    code = code[:code.index('    @blueprint.route("/workbench/experiment/activate"')]
    assert 'except WorkbenchConflict as exc:' in code
    assert 'error, status = str(exc), 409' in code
    assert 'error, status = str(exc), 400' in code
    assert '"context changed" in error.lower()' not in code


def test_stale_selection_returns_conflict_without_implicit_activation(tmp_path):
    flask = pytest.importorskip('flask')
    from ml_lab.experiments import (ExperimentStore, get_workbench_instance,
                                    select_workbench_dataset)
    from ml_lab.ui import create_ui_blueprint
    from ml_lab.experiments.datasets import describe_selection

    project = tmp_path / 'project'
    project.mkdir()
    source = project / 'values.csv'
    source.write_text('x,y\n1,4\n2,5\n3,6\n', encoding='utf8')
    app = flask.Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(create_ui_blueprint(
        name='ml_uw_12', experiment_project_root=str(project)),
        url_prefix='/pah/ml-lab')
    client = app.test_client()
    first = client.get('/pah/ml-lab/api/workbench/context').get_json()
    preview = describe_selection(ExperimentStore(project), source)
    store = ExperimentStore(project)
    activated = select_workbench_dataset(store, first['instance_id'], source,
                                         expected_generation=first['generation'])
    assert activated['generation'] > first['generation']
    stale = client.post('/pah/ml-lab/workbench/dataset/select-file', data={
        'path': str(source), 'source_sha256': preview['sha256'],
        'source_dataset_id': preview['dataset_id'],
        'expected_generation': str(first['generation']),
        'features': ['x', 'y'],
    })
    assert stale.status_code == 409
    assert 'Reload current view' in stale.get_data(as_text=True)
    assert get_workbench_instance(store, first['instance_id'])['generation'] == activated['generation']


def test_shared_recovery_macro_renders_through_child_templates():
    """Jinja inheritance and imports must work without a Flask process."""
    from types import SimpleNamespace
    env = Environment(loader=FileSystemLoader(str(TEMPLATES)),
                      autoescape=select_autoescape(['html']))
    env.globals['request'] = SimpleNamespace(blueprint='ml_test', args={})
    env.globals['url_for'] = lambda endpoint, **params: (
        '/pah/ml-lab/' + endpoint.split('.')[-1]
    )
    common = dict(version='0.0', workspace_scope='', workspace_active_tool='',
                  workspace_tools=[], context={'status':'empty', 'generation':0,
                                               'validation_error':None, 'active_dataset':None},
                  error='Problem with saved workbench state', report=None)
    cases = {
        'workbench_dataset_select': dict(source_path='missing.csv', preview=None, updated=None),
        'workbench_experiment_activate': dict(experiment=None, verified=False, activated=False),
        'workbench_representation': dict(methods=[], scalings=[], method='pca',
                                          scaling='standard', raw_components='2',
                                          raw_x='1', raw_y='2', reconstruct=False),
        'workbench_autoencoder': dict(saved=False, configured=False, recent=[],
                                       diagnostics=None, fields={}),
        'workbench_comparison': dict(methods=[], scalings=[], chosen_methods=[],
                                      scaling='standard', raw_components='2',
                                      include_neural=False, max_methods=3, max_rows=500,
                                      max_features=20),
        'workbench_general_ml_tools': dict(groups={'models':[], 'ebm':[],
                                                     'optimization':[], 'generative':[]},
                                           catalog_limit=50),
    }
    for name, props in cases.items():
        page = env.get_template(f'ml_lab_ui/{name}.html').render(**common, **props)
        assert 'Problem with saved workbench state' in page, name
        assert 'role="alert"' in page, name
        assert 'Reload current view' in page, name
        assert 'href="#mlwb-main"' in page, name
