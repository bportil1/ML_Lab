"""Large source graphs use progressive display; no implicit sampling."""
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def test_graph_controller_has_complete_search_and_page_controls():
    js=(ROOT/'src/ml_lab/ui/static/ml_lab_structured_graph.js').read_text()
    html=(ROOT/'src/ml_lab/ui/templates/ml_lab_ui/workbench_structured.html').read_text()
    for token in ('next-page','previous-page','next-records','previous-records','data-graph-row-search'):
        assert token in html
    assert "all.slice(0, 80)" not in js
    assert 'No records discarded.' in js

def test_owner_no_longer_rejects_view_by_node_or_edge_totals():
    text=(ROOT/'src/ml_lab/ui/structured_graph_view.py').read_text()
    assert 'MAX_GRAPH_NODES' not in text
    assert 'MAX_GRAPH_EDGES' not in text
    assert 'graph exceeds shared viewer' not in text


def test_larger_source_file_and_tabular_records_without_artificial_ui_caps(tmp_path):
    import json
    from ml_lab.ui.structured_workbench import read_source
    large=tmp_path/'large.json'
    large.write_text(json.dumps({'payload':'x'*(2*1024*1024)}))
    obj,meta=read_source(tmp_path,'large.json','auto')
    assert meta['nodes']>=2
    csv=tmp_path/'wide.csv'
    headers=[f'field{i}' for i in range(257)]
    csv.write_text(','.join(headers)+'\n'+','.join('x' for _ in headers)+'\n')
    _,meta=read_source(tmp_path,'wide.csv','auto')
    assert meta['nodes']==258
    many=tmp_path/'many.csv'
    many.write_text('item\n'+'x\n'*10001)
    _,meta=read_source(tmp_path,'many.csv','auto')
    assert meta['rows']==10001

def test_opt_in_source_memory_guard_remains_available(tmp_path, monkeypatch):
    import pytest
    from ml_lab.structured import StructuredContractError
    from ml_lab.ui.structured_workbench import read_source
    path=tmp_path/'sample.json';path.write_text('{"x": 3}')
    monkeypatch.setenv('ML_LAB_STRUCTURED_MAX_SOURCE_BYTES','4')
    with pytest.raises(StructuredContractError,match='configured'):
        read_source(tmp_path,'sample.json','auto')


def test_original_long_node_ids_survive_display_projection():
    from ml_lab.structured import StructuredObject
    from ml_lab.ui.structured_graph_view import make_shared_graph_view
    key='n' * 350
    obj=StructuredObject.from_dict({'schema':'ml-lab.structured-object@1',
        'topology':{'kind':'undirected_graph'},
        'nodes':[{'id':key,'kind':'record','label':key,'attributes':{}}],
        'edges':[],'observations':[],'observation_unit':'graph','provenance':{}})
    assert make_shared_graph_view(obj,fingerprint='a'*64)['nodes'][0]['id']==key
