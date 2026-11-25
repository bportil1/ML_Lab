"""Single-file user journey regression for Structured Uncertainty intake."""
from pathlib import Path
import json

import pytest

from ml_lab.structured import StructuredContractError
from ml_lab.ui.structured_workbench import read_source, analyze
from ml_lab.ui.structured_source_intake import browse_project_files


def test_xml_single_file_is_validated_not_statistical_samples(tmp_path):
    path = tmp_path / 'single.xml'
    path.write_text('<root><item id="a"><score>8</score></item><item id="b"><score>9</score></item></root>')
    obj, meta = read_source(tmp_path, 'single.xml', 'auto')
    assert meta['detected_adapter'] == 'xml_file'
    assert obj.topology['kind'] == 'tree'
    assert meta['rows'] == 0
    assert 'individual_occurrence_topology' in meta['losses']
    assert any(n['attributes'].get('occurrence_count') == 2 for n in obj.nodes)
    assert 'aligned_observations' not in obj.capabilities
    assert analyze(obj, 'data', 'structure', {})[0]['support']['nodes'] >= 2
    with pytest.raises(StructuredContractError):
        analyze(obj, 'profiling', 'information', {'x': 'score'})
    selected, _ = read_source(tmp_path, path.as_posix(), 'auto', meta['sha256'])
    assert selected.to_json() == obj.to_json()
    path.write_text('<root/>')
    with pytest.raises(StructuredContractError, match='changed'):
        read_source(tmp_path, 'single.xml', 'auto', meta['sha256'])


@pytest.mark.parametrize('name,delimiter', [('records.csv', ','), ('records.tsv', '\t')])
def test_single_delimited_file_keeps_aligned_records_and_string_types(tmp_path, name, delimiter):
    path = tmp_path / name
    path.write_text(delimiter.join(('id', 'group', 'outcome')) + '\n'
                    + delimiter.join(('001', 'A', 'yes')) + '\n'
                    + delimiter.join(('002', 'A', '')) + '\n'
                    + delimiter.join(('003', 'B', 'no')) + '\n')
    obj, meta = read_source(tmp_path, name, 'auto')
    assert meta['rows'] == 3 and meta['detected_adapter'] == 'tabular'
    assert obj.observation_unit == 'record'
    assert 'aligned_observations' in obj.capabilities
    assert list(obj.observations[0]['values']) == ['id', 'group', 'outcome']
    assert obj.observations[0]['values']['id']['value'] == '001'
    assert obj.observations[1]['values']['outcome']['status'] == 'empty'
    assert len(obj.nodes) == 4 and len(obj.edges) == 3
    report, _ = analyze(obj, 'profiling', 'observations', {'variables': 'id,group,outcome'})
    assert report['support']['aligned_rows'] == 3
    assert 'not inferred' in meta['warnings'][0] or 'strings' in meta['warnings'][0]


def test_automatic_json_schema_dispatch_and_generic_fallback(tmp_path):
    root = tmp_path / 'n.json'
    root.write_text('{"a": [1, 2]}')
    src, meta = read_source(tmp_path, root.name, 'auto')
    assert meta['detected_adapter'] == 'json' and meta['rows'] == 0
    from ml_lab.structured import StructuredObject
    (tmp_path/'object.json').write_text(src.to_json())
    result, meta = read_source(tmp_path, 'object.json', 'auto')
    assert meta['detected_adapter'] == 'structured'
    assert result.to_json() == src.to_json()
    (tmp_path/'broken.json').write_bytes(b'\xff')
    with pytest.raises(StructuredContractError, match='UTF-8'):
        read_source(tmp_path, 'broken.json', 'auto')


def test_direct_file_rejects_unsafe_paths_and_bad_rows(tmp_path):
    (tmp_path/'wrong.csv').write_text('a,b\n1,2\n3\n')
    with pytest.raises(StructuredContractError, match='columns'):
        read_source(tmp_path, 'wrong.csv', 'auto')
    (tmp_path/'dup.csv').write_text('a,a\n1,2\n')
    with pytest.raises(StructuredContractError, match='unique header'):
        read_source(tmp_path, 'dup.csv', 'auto')
    (tmp_path/'evil.xml').write_text('<!DOCTYPE data [ <!ENTITY e "x"> ]><data>&e;</data>')
    with pytest.raises(StructuredContractError, match='DTD|entity'):
        read_source(tmp_path, 'evil.xml', 'auto')
    (tmp_path/'bad.xml').write_text('<root><unclosed></root>')
    with pytest.raises(StructuredContractError, match='Invalid XML'):
        read_source(tmp_path, 'bad.xml', 'auto')
    (tmp_path/'link.csv').symlink_to(tmp_path/'dup.csv')
    with pytest.raises(StructuredContractError, match='symlink'):
        read_source(tmp_path, 'link.csv', 'auto')
    with pytest.raises(StructuredContractError, match='project-relative|inside'):
        read_source(tmp_path, '../outside.csv', 'auto')
    with pytest.raises(StructuredContractError, match='project'):
        read_source(tmp_path, '/etc/passwd', 'auto')
    with pytest.raises(StructuredContractError, match='Supported'):
        read_source(tmp_path, 'file.xml.zip', 'auto')


def test_browse_local_files_only_on_request(tmp_path):
    (tmp_path/'a.json').write_text('{}')
    (tmp_path/'z.csv').write_text('x\nv\n')
    (tmp_path/'no.py').write_text('')
    (tmp_path/'.venv').mkdir()
    (tmp_path/'.venv'/'secret.json').write_text('{}')
    (tmp_path/'link.xml').symlink_to(tmp_path/'a.json')
    (tmp_path/'data').mkdir()
    (tmp_path/'data'/'deep.xml').write_text('<root/>')
    discovered = browse_project_files(tmp_path)
    assert set(discovered) == {'a.json', 'z.csv', 'data/deep.xml'}
    assert len(browse_project_files(tmp_path, max_files=1)) == 1


def test_flask_user_can_open_csv_or_xml_then_see_structure(tmp_path):
    flask = pytest.importorskip('flask')
    from ml_lab.ui import create_ui_blueprint
    app = flask.Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(create_ui_blueprint(name='singlefile',
                            experiment_project_root=str(tmp_path)), url_prefix='/ml-lab')
    client = app.test_client()
    url = '/ml-lab/workbench/structured'
    (tmp_path/'s.csv').write_text('a,b\n1,red\n2,blue\n')
    (tmp_path/'s.xml').write_text('<root><item/></root>')
    browse = client.post(url, data={'section':'data','action':'browse'})
    assert browse.status_code == 200 and b's.csv' in browse.data and b's.xml' in browse.data
    for file_name, sample in [('s.csv', b'2</strong>'), ('s.xml', b'0</strong>')]:
        response = client.post(url, data={'section': 'data', 'action': 'select', 'path': file_name, 'kind': 'auto'})
        assert response.status_code == 200, response.data[:200]
        assert b'Action unavailable' not in response.data
        assert sample in response.data
        assert client.get(url+'?section=profiling').status_code == 200
    (tmp_path/'bad.csv').write_text('a,a\n1,2\n')
    rejected = client.post(url, data={'section': 'data', 'action':'select', 'path':'bad.csv', 'kind':'auto'})
    assert rejected.status_code == 400 and b'Action unavailable' in rejected.data
