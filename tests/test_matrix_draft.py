"""S2.2 scientific grid drafts: preserve raw bytes, stable coordinates, lineage."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from ml_lab import data
from ml_lab.experiments import ExperimentStore, describe_selection


def write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_matrix_draft_is_non_destructive_and_selectable_by_experiments(tmp_path):
    project = tmp_path / "project"
    source = write(project / "data.csv", 'name,value,note\n"a,b",1.20000000000000001,"line 1"\nother,2.5,plain\n')
    before = source.read_bytes()
    sha = data.inspect_file(source).sha256
    saved = data.commit_matrix_draft(
        source, output_root=project / "analysis/data/derived", expected_sha256=sha,
        edits=[{"row": 1, "column": 1, "value": "1.230000000000000000001"},
               {"row": 2, "column": 2, "value": 'new\tvalue\n"quoted"'}],
    )
    derived = Path(saved["path"])
    assert derived != source and source.read_bytes() == before
    with derived.open(newline="", encoding="utf-8") as stream:
        result = list(csv.reader(stream))
    assert result[1] == ["a,b", "1.230000000000000000001", "line 1"]
    assert result[2] == ["other", "2.5", 'new\tvalue\n"quoted"']
    event = data.load_provenance_event(derived)
    assert event["source"]["sha256"] == sha
    assert event["derived"]["sha256"] == saved["derived_sha256"]
    assert event["recipe"]["snapshot"]["edits"][0]["row"] == 1
    assert event["changes"]["operation_count"] == 1
    selected = describe_selection(ExperimentStore(project), derived)
    assert selected["provenance"]["event_id"] == saved["provenance_event_id"]


def test_tsv_uses_stable_source_data_rows_and_preserves_format(tmp_path):
    path = write(tmp_path / "original.tsv", "x\ty\n3\talpha\n4\tbeta\n")
    saved = data.commit_matrix_draft(path, output_root=tmp_path / "out",
                                    expected_sha256=data.inspect_file(path).sha256,
                                    edits=[{"row": 2, "column": 1, "value": "changed"}])
    result = Path(saved["path"])
    assert result.suffix == ".tsv"
    assert result.read_text() == "x\ty\n3\talpha\n4\tchanged\n"


@pytest.mark.parametrize("edits", [
    [], [{"row": 0, "column": 0, "value": "8"}],
    [{"row": 1, "column": 2, "value": "8"}],
    [{"row": 1, "column": 0, "value": 8}],
    [{"row": True, "column": 0, "value": "8"}],
    [{"row": 1, "column": 0, "value": "3.1"}],
    [{"row": 1, "column": 0, "value": "9"}, {"row": 1, "column": 0, "value": "10"}],
    [{"row": 1, "column": 0, "value": "bad\x00value"}],
])
def test_bad_drafts_fail_without_writing_dataset(tmp_path, edits):
    source = write(tmp_path / "source.csv", "count,label\n2,yes\n3,no\n")
    with pytest.raises(data.MatrixDraftError):
        data.commit_matrix_draft(source, output_root=tmp_path / "results",
                                 expected_sha256=data.inspect_file(source).sha256, edits=edits)
    assert not list((tmp_path / "results").glob("*.csv"))


def test_stale_source_rejected_and_never_overwrites_original(tmp_path):
    source = write(tmp_path / "source.csv", "count,label\n2,yes\n")
    old_sha = data.inspect_file(source).sha256
    source.write_text("count,label\n3,yes\n", encoding="utf-8")
    with pytest.raises(data.MatrixDraftError, match="changed"):
        data.commit_matrix_draft(source, output_root=tmp_path / "results",
                                 expected_sha256=old_sha, edits=[{"row": 1, "column": 0, "value": "4"}])
    assert not list((tmp_path / "results").glob("*.csv"))


def test_no_change_rejected_and_source_symlink_rejected(tmp_path):
    source = write(tmp_path / "source.csv", "count,label\n2,yes\n")
    sha = data.inspect_file(source).sha256
    with pytest.raises(data.MatrixDraftError, match="No cells changed"):
        data.commit_matrix_draft(source, output_root=tmp_path / "results", expected_sha256=sha,
                                 edits=[{"row": 1, "column": 0, "value": "2"}])
    assert not list((tmp_path / "results").glob("*.csv"))
    link = tmp_path / "link.csv"
    link.symlink_to(source)
    with pytest.raises(data.MatrixDraftError, match="symlink"):
        data.commit_matrix_draft(link, output_root=tmp_path / "results", expected_sha256=sha,
                                 edits=[{"row": 1, "column": 0, "value": "5"}])


def test_unheaded_or_malformed_source_cannot_be_edited(tmp_path):
    path = write(tmp_path / "broken.csv", "a,b\n1,2\n3,4,5\n")
    with pytest.raises(data.MatrixDraftError):
        data.commit_matrix_draft(path, output_root=tmp_path / "out", expected_sha256=data.inspect_file(path).sha256,
                                 edits=[{"row": 1, "column": 0, "value": "5"}])
