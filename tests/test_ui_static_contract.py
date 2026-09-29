from pathlib import Path


def test_table_filters_use_distinct_classes_from_transform_column_filter_controls():
    """Table header inputs must not inherit transform-browser flex/pill styles."""
    static_root = Path(__file__).resolve().parents[1] / "src" / "ml_lab" / "ui" / "static"
    tables_js = (static_root / "ml_lab_tables.js").read_text(encoding="utf-8")
    source_js = (static_root / "ml_lab_source_table.js").read_text(encoding="utf-8")
    css = (static_root / "ml_lab.css").read_text(encoding="utf-8")
    compat_css = (static_root / "ml_lab_pah_compat.css").read_text(encoding="utf-8")

    assert 'filterRow.className = "table-column-filter-row"' in tables_js
    assert 'input.className = "table-column-filter"' in tables_js
    assert 'filters.className = "table-column-filter-row"' in source_js
    assert 'input.className = "table-column-filter"' in source_js
    assert ".table-column-filter-row th" in css
    assert ".table-column-filter { width: 100%" in css
    assert ".table-column-filter," in compat_css

    # These names remain reserved for the transform column-browser buttons.
    assert ".column-filter-row, .column-browser-actions" in css
    assert ".column-filter.active" in css
