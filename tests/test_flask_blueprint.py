from __future__ import annotations

import pytest

flask = pytest.importorskip("flask")

from ml_lab.flask_adapter import create_blueprint


def test_blueprint_can_be_mounted_at_nested_prefix():
    app = flask.Flask(__name__)
    app.register_blueprint(create_blueprint(), url_prefix="/pah/services/ml-lab")
    client = app.test_client()

    health = client.get("/pah/services/ml-lab/health")
    assert health.status_code == 200
    assert health.get_json()["module"] == "ml_lab"

    estimators = client.get("/pah/services/ml-lab/estimators?task=regression")
    assert estimators.status_code == 200
    assert all(item["task"] == "regression" for item in estimators.get_json()["estimators"])
