from __future__ import annotations

from pathlib import Path

from ml_lab import data
from ml_lab.application import execute_task


def test_source_table_pages_without_hiding_rows(tmp_path: Path):
    source = tmp_path / "rows.csv"
    source.write_text("id,value,label\n" + "\n".join(f"{i},{100-i},{'even' if i % 2 == 0 else 'odd'}" for i in range(123)) + "\n", encoding="utf-8")

    first = data.read_table_page(source, page=1, page_size=50)
    third = data.read_table_page(source, page=3, page_size=50)
    all_rows = data.read_table_page(source, page_size=None)

    assert first["matched_row_count"] == 123
    assert first["page_count"] == 3
    assert first["first_row"] == 1 and first["last_row"] == 50
    assert len(third["rows"]) == 23
    assert third["first_row"] == 101 and third["last_row"] == 123
    assert len(all_rows["rows"]) == 123
    assert all_rows["page_size"] is None


def test_source_table_filters_searches_and_sorts(tmp_path: Path):
    source = tmp_path / "values.tsv"
    source.write_text("name\tvalue\tgroup\na\t10\tx\nb\t2\ty\nc\t30\tx\nd\t4\ty\n", encoding="utf-8")

    filtered = data.read_table_page(source, filters={"group": "x"}, sort_column="value", sort_direction="desc")
    assert filtered["matched_row_count"] == 2
    assert [row[0] for row in filtered["rows"]] == ["c", "a"]

    searched = execute_task("data.table", {"path": str(source), "search": "b", "page_size": 50})["result"]
    assert searched["matched_row_count"] == 1
    assert searched["rows"][0][0] == "b"
