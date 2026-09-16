from __future__ import annotations

from pathlib import Path

from ml_lab import data


def _profile_by_name(profile, name: str):
    return next(column for column in profile.columns if column.name == name)


def test_profile_infers_types_quality_distributions_and_relationships(tmp_path: Path):
    source = tmp_path / "mixed.csv"
    source.write_text(
        "id,x,y,label,flag,note\n"
        "a,1,2,A,true,short\n"
        "b,2,4,A,false,hello\n"
        "c,3,6,A,true,world\n"
        "d,4,8,A,true,text\n"
        "e,5,10,A,true,words\n"
        "f,6,12,A,true,more\n"
        "g,7,14,A,true,value\n"
        "h,8,16,A,true,thing\n"
        "i,9,18,A,true,other\n"
        "j,100,200,B,true,tail\n"
        "j,100,200,B,true,tail\n",
        encoding="utf-8",
    )

    profile = data.profile_file(source, relationship_rows=0)
    payload = profile.to_record()
    assert payload["schema"] == "ml-lab.data-profile@1"
    assert payload["summary"]["profiled_row_count"] == 11
    assert payload["summary"]["duplicate_row_count"] == 1

    assert _profile_by_name(profile, "id").likely_identifier is True
    assert _profile_by_name(profile, "x").inferred_type == "integer"
    assert _profile_by_name(profile, "flag").inferred_type == "boolean"
    assert _profile_by_name(profile, "label").inferred_type == "categorical"
    assert _profile_by_name(profile, "note").inferred_type in {"categorical", "text"}

    x = _profile_by_name(profile, "x")
    assert x.numeric_summary["outlier_count"] >= 1
    assert any(issue.code == "duplicate_rows" for issue in profile.quality_issues)
    assert any(issue.code == "likely_identifier" and issue.column == "id" for issue in profile.quality_issues)

    xy = next(rel for rel in profile.relationships if {rel.left, rel.right} == {"x", "y"})
    assert xy.pearson is not None and xy.pearson > 0.99
    assert xy.spearman is not None and xy.spearman > 0.99
    assert xy.mutual_information is not None and xy.mutual_information > 0
    assert xy.normalized_mutual_information is not None and xy.normalized_mutual_information > 0.9


def test_profile_missingness_constant_imbalance_and_datetime(tmp_path: Path):
    source = tmp_path / "quality.tsv"
    source.write_text(
        "when\tconstant\tcategory\tvalue\n"
        "2026-01-01\tx\tmajor\t1\n"
        "2026-01-02\tx\tmajor\t2\n"
        "2026-01-03\tx\tmajor\t\n"
        "2026-01-04\tx\tmajor\t\n"
        "2026-01-05\tx\tminor\t\n",
        encoding="utf-8",
    )

    profile = data.profile_file(source)
    assert _profile_by_name(profile, "when").inferred_type == "datetime"
    assert _profile_by_name(profile, "constant").constant is True
    assert _profile_by_name(profile, "value").missing_rate == 0.6
    codes = {(issue.code, issue.column) for issue in profile.quality_issues}
    assert ("constant_column", "constant") in codes
    assert ("high_missingness", "value") in codes


def test_profile_uses_deterministic_reservoir_and_reports_sampling(tmp_path: Path):
    source = tmp_path / "large.csv"
    lines = ["id,value"] + [f"row-{index},{index}" for index in range(250)]
    source.write_text("\n".join(lines) + "\n", encoding="utf-8")

    first = data.profile_file(source, max_rows=25, relationship_rows=10, random_state=7)
    second = data.profile_file(source, max_rows=25, relationship_rows=10, random_state=7)
    assert first.sampled is True
    assert first.source_row_count == 250
    assert first.profiled_row_count == 25
    assert first.to_record() == second.to_record()
    assert any("reservoir sample" in warning for warning in first.warnings)


def test_profile_paths_keeps_inventory_and_profiles_supported_files(tmp_path: Path):
    (tmp_path / "a.csv").write_text("x,y\n1,2\n2,4\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("not tabular", encoding="utf-8")

    collection = data.profile_paths([tmp_path])
    payload = collection.to_record()
    assert payload["schema"] == "ml-lab.data-profile-collection@1"
    assert payload["summary"]["inventory_file_count"] == 2
    assert payload["summary"]["profile_count"] == 1
    assert payload["inventory"]["summary"]["unsupported_file_count"] == 1


def test_profile_excludes_malformed_width_rows_and_preserves_directory_relative_path(tmp_path: Path):
    nested = tmp_path / "nested"
    nested.mkdir()
    source = nested / "messy.csv"
    source.write_text("a,b\n1,2\n3,4,5\n6,7\n", encoding="utf-8")

    collection = data.profile_paths([tmp_path])
    profile = collection.profiles[0]
    assert profile.relative_path == "nested/messy.csv"
    assert profile.source_row_count == 2
    assert profile.profiled_row_count == 2
    assert any(issue.code == "malformed_rows" for issue in profile.quality_issues)
    assert any("excluded 1 malformed-width" in warning for warning in profile.warnings)
