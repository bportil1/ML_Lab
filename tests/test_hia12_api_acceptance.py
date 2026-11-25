"""HIA-12 hosted Flask acceptance: opt-in, fresh SHA and read-only endpoints.

Skipped when Flask isn't installed; use the live PAH/ML_Lab environment for
this portion of acceptance, not a repository-file-only test environment.
"""
from __future__ import annotations

import pytest

from test_hia12_scientific_acceptance import source, mapping


def test_hia_endpoints_are_explicit_source_pinned_and_read_only(tmp_path):
    flask = pytest.importorskip('flask')
    from ml_lab.ui import create_ui_blueprint

    app = flask.Flask(__name__)
    app.config['TESTING'] = True
    app.register_blueprint(create_ui_blueprint(name='hia12', experiment_project_root=str(tmp_path)),
                           url_prefix='/lab')
    client = app.test_client()
    base = '/lab/workbench/structured'
    original = source().to_json()
    source_file = tmp_path / 'observed.json'
    source_file.write_text(original, encoding='utf-8')
    assert client.get(base + '/graph').status_code == 409
    assert client.post(base, data={'section': 'data', 'action': 'select',
                                   'path': 'observed.json', 'kind': 'structured'}).status_code == 200
    graph = client.get(base + '/graph')
    assert graph.status_code == 200
    revision = graph.json['view']['source_revision']
    assert client.get(base + '/hierarchy').json['available']

    root, a, b = mapping('root', 'z'), mapping('a', 'x'), mapping('b', 'w')
    leaf = mapping('leaf', 'y')
    requests = [
        ('atomic', 'explicit-atomic-statistics', dict(node_id='a', variable_id='x',
            occurrence_policy='preserve', bins=None, min_samples=2)),
        ('neighbors', 'explicit-neighbor-statistics', dict(node_id='a', variable_id='x',
            neighbors=[leaf], conditioning=None, occurrence_policy='preserve',
            bins={}, min_samples=2, min_stratum_samples=2)),
        ('branch-information', 'explicit-branch-information', dict(branch_root_id='a',
            mappings=[a, leaf], context=root, occurrence_policy='preserve',
            bins={}, min_samples=2, min_stratum_samples=2)),
        ('level-information', 'explicit-level-information', dict(level=1,
            mappings=[a,b], occurrence_policy='preserve', bins={}, min_samples=2)),
        ('conditional-information', 'explicit-conditional-level-information', dict(level=1,
            mappings=[a,b], preceding_mappings=[root], target=leaf,
            occurrence_policy='preserve', bins={}, min_samples=2, min_stratum_samples=2)),
        ('factorized-information', 'explicit-conditional-factorization', dict(level=1,
            mappings=[a,b], preceding_mappings=[root], target=None,
            assumption_confirmed=True, factorization_mode='shared_joint_prior',
            parent_sets={}, occurrence_policy='preserve', bins={}, min_samples=2,
            min_stratum_samples=2)),
        ('network-accounting', 'explicit-network-information-accounting', dict(
            mappings=[root,a,b], target=leaf, occurrence_policy='preserve',
            bins={}, min_samples=2)),
    ]
    for suffix, header, fields in requests:
        url = base + '/hierarchy/' + suffix
        body = {'source_revision': revision, **fields}
        assert client.post(url, json=body).status_code == 400, suffix
        assert client.post(url, json={**body, 'source_revision': 'sha256:'+'0'*64},
                           headers={'X-ML-Lab-HIA': header}).status_code == 409, suffix
        response = client.post(url, json=body, headers={'X-ML-Lab-HIA': header})
        assert response.status_code == 200, (suffix, response.json)
        assert response.json['source_revision'] == revision
        assert response.json['report']['source_digest'] == graph.json['binding_contract']['source_digest']
    # Analysis endpoints must not overwrite or mutate the user's structured file.
    assert source_file.read_text(encoding='utf-8') == original
    source_file.write_text('{}', encoding='utf-8')
    for suffix, header, fields in requests:
        body = {'source_revision': revision, **fields}
        assert client.post(base+'/hierarchy/'+suffix,json=body,
                           headers={'X-ML-Lab-HIA': header}).status_code == 409
