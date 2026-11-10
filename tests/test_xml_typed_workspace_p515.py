"""P5-15 UI integration contracts; Flask exercise remains optional in minimal environments."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'src/ml_lab/ui'


def test_typed_preview_is_independent_of_legacy_xml_extraction():
    template = (ROOT / 'templates/ml_lab_ui/xml_structure.html').read_text()
    script = (ROOT / 'static/ml_lab_xml_typed_preview.js').read_text()
    blueprint = (ROOT / 'blueprint.py').read_text()
    assert 'data-xml-typed-workspace' in template
    assert 'ml_lab_xml_typed_preview.js' in template
    assert 'data-xml-explorer' in template
    assert 'data-xml-typed-run' in script
    assert 'credentials: "same-origin"' in script
    assert 'textContent = JSON.stringify(result, null, 2)' in script
    assert '@blueprint.post("/data/xml/<token>/typed-preview")' in blueprint
    assert 'max_nodes_per_value=2000' in blueprint
    assert 'response.headers["Cache-Control"] = "no-store"' in blueprint


def test_typed_preview_endpoint_when_flask_installed(tmp_path):
    import html
    import re
    import pytest
    pytest.importorskip('flask')
    from ml_lab.ui import create_app

    source = tmp_path / 'typed.xml'
    source.write_text('<catalog><entry><name>A</name><tag>1</tag><tag>2</tag></entry><entry><name>B</name></entry></catalog>')
    client = create_app(config={'TESTING': True}).test_client()
    response = client.post('/data', data={'paths': str(source), 'preview_rows': '5'})
    assert response.status_code == 200
    match = re.search(rb'href="([^"]*/data/xml/[^"]+)"[^>]*>Explore XML</a>', response.data)
    assert match is not None
    explorer = html.unescape(match.group(1).decode())
    page = client.get(explorer)
    assert page.status_code == 200
    assert b'data-xml-typed-workspace' in page.data
    result = client.post(explorer + '/typed-preview', json={
        'record_root_canonical_path': '/catalog/entry',
        'variables': [{'canonical_path': '/catalog/entry/name', 'representation': 'scalar'},
                      {'canonical_path': '/catalog/entry/tag', 'representation': 'list'}],
        'max_records': 2,
    })
    assert result.status_code == 200, result.get_data(as_text=True)
    body = result.get_json()
    assert body['schema'] == 'ml-lab.xml-typed-preview@1'
    assert len(body['records']) == 2
    assert body['records'][0]['variables'][1]['count'] == 2
    assert body['records'][1]['variables'][1]['present'] is False
    assert result.headers['Cache-Control'] == 'no-store'
    invalid = client.post(explorer + '/typed-preview', json={
        'record_root_canonical_path': '/catalog/entry', 'variables': [], 'max_records': 1000,
    })
    assert invalid.status_code == 400
