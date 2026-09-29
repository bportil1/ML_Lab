from __future__ import annotations

import pytest

pytest.importorskip("flask")

from flask import Flask

from ml_lab.ui import create_app, create_ui_blueprint


def test_standalone_ui_renders_dashboard_and_capability_api():
    app = create_app(config={"TESTING": True})
    client = app.test_client()

    response = client.get("/")
    assert response.status_code == 200
    assert b"Machine Learning Lab" in response.data
    assert b"Data Lab" in response.data
    assert b">Compare</a>" in response.data
    assert b">Transform</a>" in response.data
    assert b">Provenance</a>" in response.data

    payload = client.get("/api/capabilities").get_json()
    assert payload["version"]
    assert any(record["id"] == "classification" for record in payload["capabilities"])


def test_ui_blueprint_mounts_under_host_prefix_without_route_assumptions():
    host = Flask(__name__)
    host.config["TESTING"] = True
    host.register_blueprint(create_ui_blueprint(name="mounted_ml_lab"), url_prefix="/tools/ml-lab")
    client = host.test_client()

    response = client.get("/tools/ml-lab/")
    assert response.status_code == 200
    assert b"ML Lab" in response.data
    assert b"/tools/ml-lab/static/ml_lab.css" in response.data
    assert b"/tools/ml-lab/static/pah-module-theme.css" in response.data
    assert b"/tools/ml-lab/static/ml_lab_pah_compat.css" in response.data
    assert response.data.index(b"ml_lab.css") < response.data.index(b"pah-module-theme.css") < response.data.index(b"ml_lab_pah_compat.css")
    assert b'data-pah-module-root' in response.data
    assert b'data-pah-tool-nav' in response.data

    task = client.get("/tools/ml-lab/task/clustering")
    assert task.status_code == 200
    assert b"Application payload" in task.data


def test_ui_task_executes_through_shared_application_service():
    app = create_app(config={"TESTING": True})
    client = app.test_client()
    payload = '''{
      "X": [[0, 0], [0.1, 0.1], [5, 5], [5.1, 5.1]],
      "feature_names": ["x", "y"],
      "estimators": ["kmeans"],
      "config": {"repeats": 1, "random_state": 7}
    }'''
    response = client.post("/task/clustering", data={"payload": payload})
    assert response.status_code == 200
    assert b'&quot;task&quot;: &quot;clustering&quot;' in response.data


def test_experimental_ui_is_hidden_unless_enabled():
    base = create_app(config={"TESTING": True}).test_client()
    assert base.get("/task/experimental").status_code == 404

    enabled = create_app(enable_experimental=True, config={"TESTING": True}).test_client()
    assert enabled.get("/task/experimental").status_code == 200


def test_data_lab_mounts_and_inspects_host_local_path(tmp_path):
    source = tmp_path / "data.csv"
    source.write_text("index,value\n0,10\n1,20\n", encoding="utf-8")

    host = Flask(__name__)
    host.config["TESTING"] = True
    host.register_blueprint(create_ui_blueprint(name="data_ml_lab"), url_prefix="/ml")
    client = host.test_client()

    landing = client.get("/ml/data")
    assert landing.status_code == 200
    assert b"Data intake &amp; profiling" in landing.data

    response = client.post(
        "/ml/data",
        data={"paths": str(source), "recursive": "1", "preview_rows": "20"},
    )
    assert response.status_code == 200
    assert "1 × 2".encode("utf-8") in response.data
    assert b"likely exported index column" in response.data
    assert b"ml-lab.data-inventory@1" in response.data



def test_path_autocomplete_is_available_across_data_workspaces():
    app = create_app(config={"TESTING": True})
    client = app.test_client()

    data_page = client.get("/data")
    assert data_page.status_code == 200
    assert b'data-path-autocomplete' in data_page.data
    assert b'data-path-multiline="true"' in data_page.data
    assert b'ml_lab_path_input.js' in data_page.data

    compare_page = client.get("/data/compare")
    assert compare_page.status_code == 200
    assert b'data-path-autocomplete' in compare_page.data
    assert b'data-path-multiline="true"' in compare_page.data

    provenance_page = client.get("/data/provenance")
    assert provenance_page.status_code == 200
    assert b'id="provenance-source"' in provenance_page.data
    assert b'data-path-autocomplete' in provenance_page.data
    assert b'data-path-kind="file"' in provenance_page.data

def test_data_lab_profiles_with_visible_job_state_and_persisted_profile(tmp_path):
    import time

    source = tmp_path / "profile.csv"
    source.write_text("x,y,label\n1,2,A\n2,4,A\n3,6,B\n", encoding="utf-8")

    app = create_app(config={"TESTING": True, "ML_LAB_PROFILE_OUTPUT_ROOT": str(tmp_path / "runs")})
    client = app.test_client()
    started = client.post(
        "/data/profile/start",
        data={
            "paths": str(source),
            "recursive": "1",
            "preview_rows": "20",
            "max_rows": "100000",
            "relationship_rows": "0",
            "max_relationship_columns": "25",
        },
    )
    assert started.status_code == 302
    job_url = started.headers["Location"]
    assert "/data/profile/job/" in job_url
    status_url = job_url + "/status"

    deadline = time.time() + 10
    payload = client.get(status_url).get_json()
    while payload["status"] not in {"completed", "failed"} and time.time() < deadline:
        time.sleep(0.02)
        payload = client.get(status_url).get_json()

    assert payload["status"] == "completed", payload.get("error")
    assert payload["profile_links"]
    profile = client.get(payload["profile_links"][0]["url"])
    assert profile.status_code == 200
    assert b"Persisted statistical profile" in profile.data
    assert b"Pearson/Spearman" in profile.data
    assert b"Open source" in profile.data
    assert b"Profile visual summary" in profile.data
    assert b"Missingness" in profile.data
    assert b"Numeric spread" in profile.data
    assert b"Relationship strength" in profile.data


def test_data_transform_workspace_previews_and_applies_typed_recipe(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("name,trial,label\n A ,1,aug_3\n B ,x,aug_6\n", encoding="utf-8")
    output = tmp_path / "derived.csv"
    app = create_app(config={"TESTING": True})
    client = app.test_client()

    page = client.get("/data/transform")
    assert page.status_code == 200
    assert b"Controlled transformation" in page.data

    form = {
        "source": str(source),
        "output": str(output),
        "recipe_name": "UI cleanup",
        "type_overrides": "trial=integer",
        "coerce_errors": "coerce",
        "clean_columns": "name",
        "derive_source": "label",
        "derive_target": "augmentation",
        "derive_pattern": r"aug_(\\d+)",
        "derive_group": "0",
        "preview_rows": "50",
        "mode": "preview",
    }
    preview = client.post("/data/transform", data=form)
    assert preview.status_code == 200
    assert b"Operation diagnostics" in preview.data
    assert b"coerce_types" in preview.data
    assert b"Transformation preview" in preview.data
    assert b"Changes only" in preview.data
    assert b"Affected rows" in preview.data
    assert b"Preview validated against the current submitted recipe" in preview.data
    assert b"ml_lab_transform_preview.js" in preview.data
    assert not output.exists()

    form["mode"] = "apply"
    applied = client.post("/data/transform", data=form)
    assert applied.status_code == 200
    assert b"Derived dataset written" in applied.data
    assert output.is_file()


def test_data_transform_column_browser_loads_source_metadata_without_transforming(tmp_path):
    source = tmp_path / "wide.csv"
    source.write_text(
        "id,score,status,constant\n"
        "1,1.5,A,x\n"
        "2,,B,x\n"
        "3,3.5,A,x\n",
        encoding="utf-8",
    )
    app = create_app(config={"TESTING": True})
    client = app.test_client()

    response = client.post(
        "/data/transform",
        data={"source": str(source), "preview_rows": "50", "mode": "columns"},
    )

    assert response.status_code == 200
    assert b"Choose columns" in response.data
    assert b"4 columns available to the selector" in response.data
    assert b'data-column-target="scale_columns"' in response.data
    assert b'data-column-target="filter_column"' in response.data
    assert b'data-column-single="true"' in response.data
    assert b'data-type="integer"' in response.data
    assert b'data-type="float"' in response.data
    assert b'data-missing="1"' in response.data
    assert b'data-constant="1"' in response.data
    assert b'data-column-pattern' in response.data
    assert b'data-select-pattern' in response.data
    assert b'data-exclude-pattern' in response.data
    assert b'data-deselect-filtered' in response.data
    assert b'data-invert-columns' in response.data
    assert b'data-save-group' in response.data
    assert b"cardinality" in response.data
    assert b"Operation diagnostics" not in response.data


def test_data_transform_spreadsheet_workspace_uses_source_headers_and_recipe_actions(tmp_path):
    source = tmp_path / "sheet.csv"
    source.write_text(
        "id,score,status\n"
        "1,1.5,A\n"
        "2,2.5,B\n",
        encoding="utf-8",
    )
    app = create_app(config={"TESTING": True})
    client = app.test_client()

    response = client.post(
        "/data/transform",
        data={"source": str(source), "preview_rows": "25", "mode": "columns"},
    )

    assert response.status_code == 200
    assert b"Dataset workspace" in response.data
    assert b'data-transform-sheet' in response.data
    assert b'data-sheet-column="score"' in response.data
    assert b'data-sheet-column-menu="status"' in response.data
    assert b'data-sheet-apply-action="scale"' in response.data
    assert b'data-sheet-apply-action="type"' in response.data
    assert b'data-sheet-apply-action="rename"' in response.data
    assert b'data-sheet-apply-action="filter"' in response.data
    assert b'data-sheet-apply-action="drop"' in response.data
    assert b"ml_lab_transform_sheet.js" in response.data
    assert b'data-sheet-recipe-context' in response.data
    assert b'data-sheet-recipe-links' in response.data
    assert b">1.5<" in response.data
    assert b"Operation diagnostics" not in response.data


def test_data_compare_workspace(tmp_path):
    app = create_app(config={"TESTING": True})
    client = app.test_client()
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text("id,x\n1,10\n2,20\n", encoding="utf-8")
    right.write_text("id,x\n1,10\n2,21\n", encoding="utf-8")
    response = client.post(
        "/data/compare",
        data={"paths": f"{left}\n{right}", "recursive": "1", "max_pairs": "200"},
    )
    assert response.status_code == 200
    assert b"Dataset comparison" in response.data
    assert b"same_schema_distinct_data" in response.data or b"likely_version_or_derivative" in response.data
    assert b"Pair overview" in response.data
    assert b"Schema" in response.data
    assert b"Left rows" in response.data


def test_data_provenance_workspace_traces_recorded_lineage(tmp_path):
    from ml_lab import data

    source = tmp_path / "source.csv"
    source.write_text("id,value\n1,10\n2,20\n", encoding="utf-8")
    derived = tmp_path / "derived.csv"
    data.apply_transformation(
        source,
        {"name": "UI lineage", "operations": [{"type": "filter_rows", "column": "value", "operator": "ge", "value": 20}]},
        output=derived,
    )

    app = create_app(config={"TESTING": True})
    client = app.test_client()
    response = client.post("/data/provenance", data={"source": str(derived)})

    assert response.status_code == 200
    assert b"Formal data provenance" in response.data
    assert b"Recorded transformation chain" in response.data
    assert b"authoritative" in response.data
    assert str(source.resolve()).encode("utf-8") in response.data
    assert str(derived.resolve()).encode("utf-8") in response.data


def test_data_provenance_workspace_traces_applied_transform(tmp_path):
    source = tmp_path / "source.csv"
    source.write_text("id,value\n1,10\n2,20\n", encoding="utf-8")
    derived = tmp_path / "derived.csv"

    app = create_app(config={"TESTING": True})
    client = app.test_client()

    transform = client.post(
        "/data/transform",
        data={
            "source": str(source),
            "output": str(derived),
            "recipe_name": "provenance test",
            "filter_column": "value",
            "filter_operator": "ge",
            "filter_value": "20",
            "preview_rows": "50",
            "mode": "apply",
        },
    )
    assert transform.status_code == 200
    assert derived.is_file()
    assert b"Provenance:" in transform.data
    assert b"Open lineage" in transform.data

    page = client.post("/data/provenance", data={"source": str(derived)})
    assert page.status_code == 200
    assert b"Formal data provenance" in page.data
    assert b"Recorded transformation chain" in page.data
    assert b"Hash-valid chain" in page.data
    assert b"authoritative" in page.data
    assert b"ml-lab.data-lineage@1" in page.data


def test_data_transform_recipe_panel_renders_live_pipeline_controls():
    app = create_app(config={"TESTING": True})
    client = app.test_client()
    response = client.get("/data/transform")
    assert response.status_code == 200
    assert b"Transformation Recipe" in response.data
    assert b'data-recipe-order' in response.data
    assert b'data-recipe-disabled' in response.data
    assert b"ml_lab_transform_recipe.js" in response.data


def test_transform_form_recipe_order_and_disabled_groups_are_authoritative():
    from ml_lab.ui.blueprint import _recipe_from_transform_form

    app = Flask(__name__)
    form = {
        "recipe_name": "Ordered UI recipe",
        "clean_columns": "name",
        "filter_column": "name",
        "filter_operator": "eq",
        "filter_value": "A",
        "scale_columns": "score",
        "scale_method": "minmax",
        "recipe_order": "filter_rows,clean_strings,scale",
        "recipe_disabled": "scale",
    }
    with app.test_request_context("/data/transform", method="POST", data=form):
        recipe = _recipe_from_transform_form()

    assert [operation["type"] for operation in recipe["operations"]] == ["filter_rows", "clean_strings"]


def test_transform_path_autocomplete_is_local_root_scoped_and_filters_extensions(tmp_path):
    source_dir = tmp_path / "datasets"
    source_dir.mkdir()
    csv_path = source_dir / "sample.csv"
    csv_path.write_text("x\n1\n", encoding="utf-8")
    tsv_path = source_dir / "sample.tsv"
    tsv_path.write_text("x\n1\n", encoding="utf-8")
    (source_dir / "sample.txt").write_text("ignore", encoding="utf-8")
    nested = source_dir / "nested"
    nested.mkdir()

    host = Flask(__name__)
    host.config["TESTING"] = True
    host.register_blueprint(
        create_ui_blueprint(name="paths_ml_lab", path_autocomplete_roots=[tmp_path]),
        url_prefix="/ml",
    )
    client = host.test_client()

    response = client.get(
        "/ml/api/path-suggestions",
        query_string={"q": str(source_dir / "sam"), "kind": "file", "extensions": "csv,tsv"},
    )
    assert response.status_code == 200
    payload = response.get_json()
    paths = {item["path"] for item in payload["suggestions"]}
    assert str(csv_path) in paths
    assert str(tsv_path) in paths
    assert str(source_dir / "sample.txt") not in paths

    directories = client.get(
        "/ml/api/path-suggestions",
        query_string={"q": str(source_dir) + "/", "kind": "directory"},
    ).get_json()["suggestions"]
    assert {item["path"] for item in directories} == {str(nested)}

    outside = client.get(
        "/ml/api/path-suggestions",
        query_string={"q": str(tmp_path.parent) + "/", "kind": "either"},
    )
    assert outside.status_code == 200
    assert outside.get_json()["suggestions"] == []


def test_transform_page_uses_reusable_path_autocomplete_component(tmp_path):
    app = create_app(
        config={
            "TESTING": True,
            "ML_LAB_PATH_AUTOCOMPLETE_ROOTS": [str(tmp_path)],
        }
    )
    response = app.test_client().get("/data/transform")
    assert response.status_code == 200
    assert b'data-path-autocomplete' in response.data
    assert b'data-path-extensions="csv,tsv,xml"' in response.data
    assert b'ml_lab_path_input.js' in response.data
    assert b'/api/path-suggestions' in response.data


def test_xml_structure_explorer_renders_hierarchy_and_inspector(tmp_path):
    import html
    import re

    source = tmp_path / "catalog.xml"
    source.write_text(
        '''<?xml version="1.0" encoding="UTF-8"?>
<catalog xmlns:m="urn:metrics">
  <project id="alpha"><name>A</name><m:metrics><m:cwe id="190" count="15"/><m:cwe id="191" count="7"/></m:metrics></project>
  <project id="beta"><name>B</name><m:metrics><m:cwe id="190" count="6"/></m:metrics></project>
</catalog>
''',
        encoding="utf-8",
    )

    app = create_app(config={"TESTING": True})
    client = app.test_client()
    inventory = client.post(
        "/data",
        data={"paths": str(source), "recursive": "1", "preview_rows": "20"},
    )
    assert inventory.status_code == 200
    assert b"Explore XML" in inventory.data
    match = re.search(rb'href="([^"]*/data/xml/[^"]+)"[^>]*>Explore XML</a>', inventory.data)
    assert match is not None

    explorer_url = html.unescape(match.group(1).decode("utf-8"))
    response = client.get(explorer_url)

    assert response.status_code == 200
    assert b"XML structural analysis" in response.data
    assert b"Explore the XML hierarchy" in response.data
    assert b'data-xml-explorer' in response.data
    assert b'data-xml-tree' in response.data
    assert b'data-xml-inspector' in response.data
    assert b'data-xml-candidates-only' in response.data
    assert b"Candidate record roots" in response.data
    assert b"urn:metrics" in response.data
    assert b"/catalog/project" in response.data
    assert b"record candidate" in response.data
    assert b"ml_lab_xml_structure.js" in response.data
    assert b"ml-lab.xml-structure@2" in response.data


def test_xml_structure_explorer_keeps_raw_xml_non_tabular(tmp_path):
    import html
    import re

    source = tmp_path / "simple.xml"
    source.write_text("<root><item id='1'/><item id='2'/></root>", encoding="utf-8")
    app = create_app(config={"TESTING": True})
    client = app.test_client()

    inventory = client.post("/data", data={"paths": str(source), "recursive": "1", "preview_rows": "20"})
    match = re.search(rb'href="([^"]*/data/xml/[^"]+)"[^>]*>Explore XML</a>', inventory.data)
    assert match is not None
    response = client.get(html.unescape(match.group(1).decode("utf-8")))

    assert response.status_code == 200
    assert b"does not flatten or transform the source" in response.data
    assert b"Transform this source" not in response.data
    assert b"Recommendations only. You choose the record root explicitly." in response.data


def test_xml_structure_explorer_rejects_malformed_xml_from_direct_inventory_action(tmp_path):
    source = tmp_path / "broken.xml"
    source.write_text("<root><item></root>", encoding="utf-8")
    app = create_app(config={"TESTING": True})
    client = app.test_client()

    inventory = client.post("/data", data={"paths": str(source), "recursive": "1", "preview_rows": "20"})
    assert inventory.status_code == 200
    assert b"Explore XML" not in inventory.data
    assert b"ParseError" in inventory.data


def test_xml_record_root_field_selector_endpoint_and_controls(tmp_path):
    import html
    import re

    source = tmp_path / "records.xml"
    source.write_text(
        "<catalog><project id='a'><name>A</name></project><project id='b'><name>B</name></project></catalog>",
        encoding="utf-8",
    )
    app = create_app(config={"TESTING": True})
    client = app.test_client()
    inventory = client.post("/data", data={"paths": str(source), "recursive": "1", "preview_rows": "20"})
    match = re.search(rb'href="([^"]*/data/xml/[^"]+)"[^>]*>Explore XML</a>', inventory.data)
    assert match is not None
    explorer_url = html.unescape(match.group(1).decode("utf-8"))
    page = client.get(explorer_url)

    assert page.status_code == 200
    assert b'data-xml-field-selector' in page.data
    assert b'data-xml-use-record-root' in page.data
    assert b'data-xml-choose-record-root' in page.data
    assert b'Extraction configuration' in page.data
    fields_url_match = re.search(rb'data-record-fields-url="([^"]+)"', page.data)
    assert fields_url_match is not None
    fields_url = html.unescape(fields_url_match.group(1).decode("utf-8"))

    response = client.get(fields_url, query_string={"root": "/catalog/project"})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["schema"] == "ml-lab.xml-record-selection@1"
    assert payload["record_root"]["path"] == "/catalog/project"
    relative = {field["relative_path"] for field in payload["fields"]}
    assert {"@id", "name"}.issubset(relative)

    invalid = client.get(fields_url, query_string={"root": "/catalog/missing"})
    assert invalid.status_code == 400
    assert "not present" in invalid.get_json()["error"]


def test_xml_repeated_branch_rule_endpoint_and_controls(tmp_path):
    import html
    import re

    source = tmp_path / "rules.xml"
    source.write_text(
        "<catalog><project id='a'><cwe id='190' count='2'/><cwe id='191' count='1'/></project>"
        "<project id='b'><cwe id='190' count='3'/></project></catalog>",
        encoding="utf-8",
    )
    app = create_app(config={"TESTING": True})
    client = app.test_client()
    inventory = client.post("/data", data={"paths": str(source), "recursive": "1", "preview_rows": "20"})
    match = re.search(rb'href="([^"]*/data/xml/[^"]+)"[^>]*>Explore XML</a>', inventory.data)
    assert match is not None
    explorer_url = html.unescape(match.group(1).decode("utf-8"))
    page = client.get(explorer_url)

    assert page.status_code == 200
    assert b"Repeated branch rules" in page.data
    assert b"One-to-many rules" in page.data
    assert b'data-xml-rule-panel' in page.data
    plan_url_match = re.search(rb'data-collection-plan-url="([^"]+)"', page.data)
    assert plan_url_match is not None
    plan_url = html.unescape(plan_url_match.group(1).decode("utf-8"))

    fields_url_match = re.search(rb'data-record-fields-url="([^"]+)"', page.data)
    fields_url = html.unescape(fields_url_match.group(1).decode("utf-8"))
    fields_response = client.get(fields_url, query_string={"root": "/catalog/project"})
    fields = fields_response.get_json()["fields"]
    selected = [field["field_id"] for field in fields if field["relative_path"] in {"@id", "cwe/@id", "cwe/@count"}]

    discovery = client.post(
        plan_url,
        json={"record_root_canonical_path": "/catalog/project", "selected_field_ids": selected, "rules": []},
    )
    assert discovery.status_code == 200
    payload = discovery.get_json()
    assert payload["schema"] == "ml-lab.xml-collection-plan@1"
    assert payload["summary"]["repeated_branch_count"] == 1
    assert payload["summary"]["unresolved_branch_count"] == 1
    branch = payload["repeated_branches"][0]
    assert branch["relative_path"] == "cwe"
    assert {item["id"] for item in payload["strategy_catalog"]} >= {"pivot", "explode_rows", "separate_table"}

    configured = client.post(
        plan_url,
        json={
            "record_root_canonical_path": "/catalog/project",
            "selected_field_ids": selected,
            "rules": [{"branch_canonical_path": branch["canonical_path"], "strategy": "join", "options": {"join_delimiter": "; "}}],
        },
    )
    assert configured.status_code == 200
    configured_payload = configured.get_json()
    assert configured_payload["summary"]["ready_for_preview"] is True
    assert configured_payload["repeated_branches"][0]["rule"]["strategy"] == "join"
    assert configured_payload["repeated_branches"][0]["rule"]["options"]["join_delimiter"] == "; "
