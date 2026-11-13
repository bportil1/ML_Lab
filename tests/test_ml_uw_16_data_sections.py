"""ML-UW-16: CSV/TSV-only Tabular Data, XML Data and Postcollection Utilities.

All headless tests remain executable when optional Flask is absent.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from jinja2 import Environment, FileSystemLoader

from ml_lab import data
from ml_lab.application import execute_task


TEMPLATES = Path(__file__).resolve().parents[1] / 'src/ml_lab/ui/templates'
UI = Path(__file__).resolve().parents[1] / 'src/ml_lab/ui'


def _sources(root: Path):
    (root / 'values.csv').write_text('x,y\n1,2\n3,4\n', encoding='utf-8')
    (root / 'values.tsv').write_text('x\ty\n5\t6\n', encoding='utf-8')
    (root / 'source.xml').write_text('<dataset><record><value>1</value></record></dataset>', encoding='utf-8')


def test_tabular_intake_never_parses_xml(tmp_path):
    _sources(tmp_path)
    filtered = data.inspect_paths([tmp_path], allowed_suffixes=('.csv', '.tsv'))
    assert {Path(item.path).suffix for item in filtered.files} == {'.csv', '.tsv'}
    assert filtered.parsed_file_count == 2
    assert all(item.format == 'delimited_text' for item in filtered.files)
    default = data.inspect_paths([tmp_path])
    assert any(item.format == 'xml' for item in default.files)  # Existing CLI/API behavior retained.

    with pytest.raises(ValueError, match='Tabular Data accepts CSV/TSV'):
        data.inspect_paths([tmp_path / 'source.xml'], allowed_suffixes=('.csv', '.tsv'))
    profile = data.profile_paths([tmp_path], allowed_suffixes=('.csv', '.tsv'),
                                 relationship_rows=0, max_relationship_pairs=0)
    assert {Path(item.path).suffix for item in profile.profiles} == {'.csv', '.tsv'}


def test_tabular_application_task_filters_before_xml_parsing(tmp_path, monkeypatch):
    _sources(tmp_path)
    import ml_lab.data.intake as intake
    def prohibited(*args, **kwargs):
        raise AssertionError('XML parser invoked from Tabular Data')
    monkeypatch.setattr(intake, '_inspect_xml', prohibited)
    result = execute_task('data.inspect', {
        'paths': [str(tmp_path)], 'allowed_suffixes': ['.csv', '.tsv'],
    })['result']
    assert result['summary']['discovered_file_count'] == 2


def test_xml_c2_handles_more_than_100_parsed_sources_without_count_cap(tmp_path):
    for index in range(102):
        (tmp_path / f'file{index:03d}.xml').write_text(
            '<rows><item id="a"><value>1</value></item>'
            '<item id="b"><value>2</value></item></rows>', encoding='utf-8')
    inventory = data.inventory_xml_collection(tmp_path, max_files=150)
    assert len(inventory.files) == 102
    report = data.compare_xml_collection(inventory).to_record()
    assert report['summary']['compared_source_count'] == 102
    assert report['summary']['pair_count'] == 102 * 101 // 2
    assert report['summary']['group_count'] == 1
    with pytest.raises(ValueError, match='source limit'):
        data.compare_xml_collection(inventory, max_sources=101)  # Explicit opt-in budget retained.


def test_workspace_templates_separate_three_sections():
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=True)
    for file in (TEMPLATES / 'ml_lab_ui').glob('*.html'):
        env.get_template('ml_lab_ui/' + file.name)
    tabular = (TEMPLATES / 'ml_lab_ui/data.html').read_text()
    assert 'data-path-extensions="csv,tsv"' in tabular
    assert 'xml_collection_inventory' not in tabular
    assert 'xml_workbench' not in tabular
    assert 'xml_structure_url(file.path)' not in tabular
    xml = (TEMPLATES / 'ml_lab_ui/xml_workbench.html').read_text()
    assert 'XML Data · unified XML workflow' in xml
    assert 'xml_collection_review' in xml
    comparison = (TEMPLATES / 'ml_lab_ui/xml_collection_inventory.html').read_text()
    assert 'maximum 100 parsed sources' not in comparison
    nav = (UI / 'workbench_navigation.py').read_text()
    for label in ('Tabular Data', 'XML Data', 'Postcollection Utilities'):
        assert label in nav
    assert 'workbench_postcollection' in nav


def test_mounted_workbench_routes_and_explicit_xml_handoff(tmp_path):
    flask = pytest.importorskip('flask')
    from ml_lab.ui import create_ui_blueprint
    from ml_lab.ui.workbench_navigation import section_for_request

    _sources(tmp_path)
    app = flask.Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(create_ui_blueprint(name='uw16', experiment_project_root=str(tmp_path)),
                           url_prefix='/pah/ml')
    client = app.test_client()
    assert section_for_request('uw16.xml_collection_inventory', None) == 'xml'
    assert section_for_request('uw16.data_lab', None) == 'tabular'
    assert section_for_request('uw16.data_provenance', None) == 'postcollection'
    for route, title in (
        ('/data', 'Tabular Data'),
        ('/data/xml/workbench', 'XML Data'),
        ('/workbench/postcollection', 'Postcollection Utilities'),
    ):
        response = client.get('/pah/ml' + route)
        assert response.status_code == 200, route
        html = response.get_data(as_text=True)
        assert title in html
        assert html.count('class="mlwb-primary-link') == 7
    tabular = client.post('/pah/ml/data', data={'paths': str(tmp_path), 'recursive': '1'})
    assert tabular.status_code == 200
    assert b'values.csv' in tabular.data
    assert b'source.xml' not in tabular.data
    rejected = client.post('/pah/ml/data', data={'paths': str(tmp_path / 'source.xml')})
    assert b'Tabular Data accepts CSV/TSV' in rejected.data
    xml = client.post('/pah/ml/data/xml/collection', data={
        'directory': str(tmp_path), 'recursive': '1', 'compare': '1',
    })
    assert xml.status_code == 200
    assert b'Compared XML sources' in xml.data
