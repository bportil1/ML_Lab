"""ML-UW-02 checks that run without the optional Flask UI extra."""
from pathlib import Path

from jinja2 import Environment, FileSystemLoader


UI = Path(__file__).resolve().parents[1] / 'src' / 'ml_lab' / 'ui'


def test_workbench_templates_parse_without_server():
    environment = Environment(loader=FileSystemLoader(UI / 'templates'))
    for template in ('base.html', 'workbench_section.html', 'index.html'):
        environment.get_template('ml_lab_ui/' + template)


def test_sidebar_order_and_no_legacy_horizontal_bar():
    contract = (UI / 'workbench_navigation.py').read_text(encoding='utf-8')
    positions = [contract.index('("' + key + '", "' + title + '"') for key, title in (
        ('tabular', 'Tabular Data'), ('xml', 'XML Data'),
        ('postcollection', 'Postcollection Utilities'), ('profiling', 'Feature Profiling'),
        ('representation', 'Representation'), ('comparison', 'Comparison'),
        ('tools', 'General ML Tools'))]
    assert positions == sorted(positions)
    template = (UI / 'templates' / 'ml_lab_ui' / 'base.html').read_text(encoding='utf-8')
    assert '<nav class="module-nav' not in template
    assert 'mlwb-primary-nav' in template
    assert 'any order' not in template.lower()
    assert 'data-mlwb-context-url' in template
    assert 'data-workspace-tool="{{ tool.id }}"' in template
    assert 'data-workspace-scope="{{ workspace_scope }}"' in template
    assert "filename='ml_lab_workspace.js'" in template
    assert '<nav class="scientific-tool-list' not in template


def test_noreverse_implementation_is_not_spoofed():
    # The route additions are navigation only; neither inverse transforms nor
    # fake reconstruction operations may appear in the section implementation.
    section = (UI / 'workbench_navigation.py').read_text(encoding='utf-8')
    assert 'inverse_transform(' not in section
    assert 'Planned' in section
    assert 'install_workbench_navigation' in section
