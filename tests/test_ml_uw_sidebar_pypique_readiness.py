"""Sidebar-only visual and state regression; no new ML operations required."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from jinja2 import Environment, FileSystemLoader

UI = Path(__file__).resolve().parents[1] / 'src/ml_lab/ui'


def _render(active: str, *, subpanels: bool):
    env = Environment(loader=FileSystemLoader(UI / 'templates'))
    request = SimpleNamespace(blueprint='ml_lab', endpoint=f'ml_lab.{active}',
                              args={}, values={}, view_args={})
    labels = ('Data', 'Feature Profiling', 'Representation', 'Comparison', 'General ML Tools')
    ids = ('data', 'profiling', 'representation', 'comparison', 'tools')
    navigation = {
        'sections': [{'id': id, 'label': label, 'url': f'/ml/{id}', 'active': id == active}
                     for id, label in zip(ids, labels)],
        'data_subsections': [{'id': id, 'label': label, 'url': f'/ml/data/{id}', 'active': id == 'xml'}
                             for id, label in [('tabular','Tabular Data'), ('xml','XML Data'),
                                               ('postcollection','Post-Collection Utilities')]] if subpanels else [],
        'active_label': next((label for id, label in zip(ids, labels) if id == active), 'Tools'),
        'links': [{'url':f'/ml/{active}/analysis','label':'Structured uncertainty','current':False}]
                 if active != 'data' else [],
        'tools': [{'url':'/ml/existing','title':'Existing scientific tool','id':'old',
                   'endpoint':'data_lab'}] if active != 'tools' else [],
    }
    return env.get_template('ml_lab_ui/base.html').render(
        version='0.32.0', workspace_scope='project', workspace_active_tool='',
        workspace_tools=[], mlwb_navigation=navigation, request=request,
        url_for=lambda endpoint, **kwargs: (f'/ml/static/{kwargs["filename"]}'
                                              if "filename" in kwargs else
                                              f'/ml/{endpoint.split(".")[-1]}'),
    )


def test_five_stages_are_linked_and_have_one_readiness_line():
    html = _render('data', subpanels=True)
    assert html.count('class="mlwb-primary-link') == 5
    assert html.count('data-mlwb-stage=') == 5
    assert html.count('data-mlwb-step-summary') == 5
    assert html.count('aria-describedby="mlwb-stage-summary-') == 5
    for id in ('data','profiling','representation','comparison','tools'):
        assert f'data-mlwb-stage="{id}"' in html
        assert f'id="mlwb-stage-summary-{id}"' in html
    assert 'data-mlwb-dataset-summary' in html
    assert 'ml_lab_pypique_sidebar.css' in html
    assert 'data-mlwb-preflight-url' in html
    assert 'class="mlwb-secondary-nav mlwb-data-subsections"' in html
    assert 'class="mlwb-primary-link active"' in html
    assert 'aria-current="page"' in html
    for text in ('Tabular Data', 'XML Data', 'Post-Collection Utilities'):
        assert f'>{text}</a>' in html
    assert html.index('>General ML Tools</a>') < html.index('aria-label="Data subsections"')


def test_non_data_subpanels_remain_present_and_data_subpanel_does_not_leak():
    html = _render('representation', subpanels=False)
    assert 'aria-label="Representation options"' in html
    assert '>Structured uncertainty</a>' in html
    assert 'aria-label="Existing scientific tools"' in html
    assert '>Existing scientific tool</a>' in html
    assert 'aria-label="Data subsections"' not in html
    assert html.count('data-mlwb-stage=') == 5


def test_sidebar_styles_do_not_touch_sections_or_workbench_operations():
    css = (UI / 'static/ml_lab_pypique_sidebar.css').read_text()
    template = (UI / 'templates/ml_lab_ui/base.html').read_text()
    script = (UI / 'static/ml_lab_pypique_preflight.js').read_text()
    assert '.mlwb-nav-stage' in css
    assert '.mlwb-stage-summary' in css
    assert '.mlwb-data-subsections' in css
    assert 'data-mlwb-dataset-name' in template
    assert 'data-mlwb-step-summary' in template
    assert 'displayStage(step, data)' in script
    assert 'displayUnavailable()' in script
    assert 'fetch(panel.dataset.mlwbPreflightUrl' in script
    assert 'No active dataset — checking saved work' in script
    assert 'selected source' not in script.lower() or 'changed' in script.lower()


def test_sidebar_readiness_source_is_valid_javascript_and_scenarios_are_verified():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node.js not available for executable DOM read-only readiness tests')
    spec = Path(__file__).resolve().parent / 'js/test_ml_lab_sidebar_readiness.cjs'
    result = subprocess.run([node, str(spec)], check=False, capture_output=True,
                            text=True, timeout=20)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '6 states passed' in result.stdout
