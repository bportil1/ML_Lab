"""Regression: explicit XML review budget is not silently clamped to 80."""
from pathlib import Path

import pytest

from ml_lab.data.xml_collection import inventory_xml_collection
from ml_lab.data.xml_similarity import compare_xml_collection
from ml_lab.data.xml_collection_review import review_xml_collection
from ml_lab.data.xml_collection_recipe import build_xml_collection_recipe
from ml_lab.data.xml_collection_export import (
    XmlCollectionExportError, _verify_recipe, preview_xml_collection_export,
)


def _large_compatible_collection(root: Path, count: int = 82):
    for index in range(count):
        (root / f"source_{index:03d}.xml").write_text(
            f"<rows><item><name>entry-{index}</name></item></rows>", encoding="utf-8"
        )
    inventory = inventory_xml_collection(root, max_files=count)
    comparison = compare_xml_collection(inventory)
    selected = sorted(item.relative_path for item in inventory.files)
    return inventory, comparison, selected


def test_collection_review_recipe_and_preview_accept_more_than_eighty(tmp_path: Path):
    inventory, comparison, selected = _large_compatible_collection(tmp_path)
    assert len(selected) == 82
    assert len(comparison.pairs) == 82 * 81 // 2
    review = review_xml_collection(comparison, selected)
    assert review['status'] == 'ready_for_recipe_design'
    recipe = build_xml_collection_recipe(
        inventory, comparison, selected, '/rows/item', ['text:/rows/item/name']
    )
    assert recipe['status'] == 'ready_for_collection_preview'
    assert len(recipe['selected_sources']) == 82
    # XML-C5/C6/C7 no longer impose an unrelated default 80-source ceiling.
    checked, _ = _verify_recipe(recipe, max_files=None, max_source_bytes=64 * 1024**2)
    assert len(checked['selected_sources']) == 82
    preview = preview_xml_collection_export(recipe)
    assert len(preview['sources']) == 82
    assert preview['no_files_written']
    with pytest.raises(XmlCollectionExportError, match='explicitly requested file limit'):
        _verify_recipe(recipe, max_files=80, max_source_bytes=64 * 1024**2)


def test_review_ui_does_not_clamp_or_silently_limit_sources(tmp_path: Path):
    flask = pytest.importorskip('flask')
    from ml_lab.ui import create_app

    _large_compatible_collection(tmp_path)
    client = create_app(config={'TESTING': True}).test_client()
    params = {'directory': str(tmp_path), 'max_files': '82', 'recursive': '1'}
    response = client.post('/data/xml/collection/review', data=params)
    assert response.status_code == 200
    assert b'82</strong><span>Analyzed XML files' in response.data
    assert b'limited to 80 XML files' not in response.data
    # The same limit must survive transition into C4.
    assert b'name="max_files" value="82"' in response.data
    recipe = client.post('/data/xml/collection/recipe', data={
        **params, 'mode': 'inspect', 'selected': ['source_000.xml', 'source_001.xml'],
    })
    assert recipe.status_code == 200
