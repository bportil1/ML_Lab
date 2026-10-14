"""XML-C3 review contract: selection is evidence, never implicit merge approval."""
from __future__ import annotations

from pathlib import Path

import pytest

from ml_lab import data
from ml_lab.data.xml_collection_review import review_xml_collection


def _collection(folder: Path):
    (folder / "a.xml").write_text('<rows><item id="1"><name>A</name></item></rows>')
    (folder / "b.xml").write_text('<rows><item id="2"><name>B</name></item></rows>')
    (folder / "variant.xml").write_text('<rows><item id="3"><name>C</name><note>X</note></item></rows>')
    (folder / "other.xml").write_text('<other><thing/></other>')
    (folder / "broken.xml").write_text('<broken>')
    return data.compare_xml_collection(data.inventory_xml_collection(folder))


def test_exact_membership_is_ready_for_recipe_only(tmp_path: Path):
    report = _collection(tmp_path)
    review = review_xml_collection(report, ["a.xml", "b.xml"])
    assert review["status"] == "ready_for_recipe_design"
    assert review["summary"]["exact_pair_count"] == 1
    assert review["merge_authorized"] is False
    assert review["extraction_recipe_validated"] is False
    assert all(source["source_fingerprint"].startswith("sha256:") for source in review["selected"])
    assert review["summary"]["omitted_count"] == 3


def test_variants_need_acknowledgment_and_incompatible_stays_blocked(tmp_path: Path):
    report = _collection(tmp_path)
    review = review_xml_collection(report, ["a.xml", "variant.xml"])
    assert review["status"] == "blocked"
    assert "acknowledgment" in " ".join(review["blocking_reasons"])
    approved = review_xml_collection(report, ["a.xml", "variant.xml"], acknowledge_variants=True)
    assert approved["status"] == "ready_for_recipe_design"
    assert approved["summary"]["variant_pair_count"] == 1
    assert approved["merge_authorized"] is False
    blocked = review_xml_collection(report, ["a.xml", "other.xml"], acknowledge_variants=True)
    assert blocked["status"] == "blocked"
    assert blocked["summary"]["incompatible_pair_count"] == 1


def test_invalid_selected_source_or_missing_pair_not_approved(tmp_path: Path):
    report = _collection(tmp_path)
    assert review_xml_collection(report, ["a.xml", "broken.xml"])["status"] == "blocked"
    assert review_xml_collection(report, ["a.xml"])["status"] == "blocked"
    with pytest.raises(ValueError, match="unknown"):
        review_xml_collection(report, ["a.xml", "absent.xml"])
    with pytest.raises(ValueError, match="duplicate"):
        review_xml_collection(report, ["a.xml", "a.xml"])
    copy = {**report, "pairwise": []}
    assert review_xml_collection(copy, ["a.xml", "b.xml"])["status"] == "blocked"


def test_ui_review_and_download_json_not_csv(tmp_path: Path):
    pytest.importorskip("flask")
    from ml_lab.ui import create_app

    _collection(tmp_path)
    client = create_app(config={"TESTING": True}).test_client()
    assert client.get("/data/xml/collection/review").status_code == 200
    assert client.get("/data/xml/collection").status_code == 200
    settings = {"directory": str(tmp_path), "recursive": "1", "max_files": "40"}
    response = client.post("/data/xml/collection/review", data=settings)
    assert response.status_code == 200
    assert b"Exact structure groups" in response.data
    assert b"variant.xml" in response.data
    assert b"broken.xml" in response.data
    assert b"Differences / evidence" in response.data
    assert b"Choose collection membership" in response.data
    assert b"Explore XML" in response.data
    proposed = {**settings, "mode": "review", "selected": ["a.xml", "variant.xml"]}
    response = client.post("/data/xml/collection/review", data=proposed)
    assert b"Selection review: blocked" in response.data
    response = client.post("/data/xml/collection/review", data={**proposed, "mode": "export"})
    assert response.status_code == 400
    response = client.post("/data/xml/collection/review", data={**proposed, "mode": "export", "acknowledge_variants": "1"})
    assert response.status_code == 200
    assert response.mimetype == "application/json"
    assert response.get_json()["status"] == "ready_for_recipe_design"
    assert not response.get_json()["merge_authorized"]
    assert "attachment" in response.headers["Content-Disposition"]
    assert response.headers["Cache-Control"] == "no-store"


def test_review_validates_request_bounds_and_modified_sources(tmp_path: Path):
    pytest.importorskip("flask")
    from ml_lab.ui import create_app

    _collection(tmp_path)
    client = create_app(config={"TESTING": True}).test_client()
    url = "/data/xml/collection/review"
    settings = {"directory": str(tmp_path), "recursive": "1"}
    capped = client.post(url, data={**settings, "max_files": "1000"})
    assert capped.status_code == 200
    assert b"limited to 80 XML files" in capped.data
    assert client.post(url, data={**settings, "min_shared_fraction": "nan"}).status_code == 400
    assert client.post(url, data={**settings, "mode": "review", "selected": ["bogus.xml"]}).status_code == 400
