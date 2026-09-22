from pathlib import Path

from ml_lab import data
from ml_lab.application import execute_task


def test_exact_duplicate_and_subset_relationships(tmp_path: Path):
    left = tmp_path / "a.csv"
    duplicate = tmp_path / "a_copy.csv"
    superset = tmp_path / "superset.csv"
    text = "id,value\n1,10\n2,20\n"
    left.write_text(text, encoding="utf-8")
    duplicate.write_text(text, encoding="utf-8")
    superset.write_text("id,value\n1,10\n2,20\n3,30\n", encoding="utf-8")

    exact = data.compare_files(left, duplicate)
    assert exact["relationship"]["label"] == "exact_duplicate"
    assert exact["relationship"]["exact_duplicate"] is True

    subset = data.compare_files(left, superset)
    assert subset["relationship"]["label"] == "left_subset_of_right"
    assert subset["row_comparison"]["left_overlap_fraction"] == 1.0


def test_train_test_and_version_heuristics(tmp_path: Path):
    train = tmp_path / "project_train.csv"
    test = tmp_path / "project_test.csv"
    version = tmp_path / "project_v2.csv"
    train.write_text("id,x,y\n1,1,10\n2,2,20\n3,3,30\n4,4,40\n", encoding="utf-8")
    test.write_text("id,x,y\n10,5,50\n11,6,60\n", encoding="utf-8")
    version.write_text("id,x,y\n1,1,10\n2,2,20\n3,3,31\n4,4,40\n5,5,50\n", encoding="utf-8")

    split = data.compare_files(train, test)
    assert split["relationship"]["label"] == "likely_train_test_split"
    assert split["schema_comparison"]["same_schema"] is True

    related = data.compare_files(train, version)
    assert related["relationship"]["label"] == "likely_version_or_derivative"
    assert related["identifier_comparison"]["column"] == "id"


def test_compare_paths_and_application_contract(tmp_path: Path):
    (tmp_path / "one.csv").write_text("a,b\n1,2\n", encoding="utf-8")
    (tmp_path / "two.csv").write_text("a,b\n3,4\n", encoding="utf-8")
    (tmp_path / "three.csv").write_text("a,c\n5,6\n", encoding="utf-8")

    result = data.compare_paths([tmp_path], max_pairs=2)
    assert result["schema"] == "ml-lab.dataset-comparison-collection@1"
    assert result["summary"]["dataset_count"] == 3
    assert result["summary"]["compared_pair_count"] == 2
    assert result["summary"]["truncated"] is True

    app = execute_task("data.compare", {"paths": [str(tmp_path)], "max_pairs": 0})
    assert app["task"] == "data.compare"
    assert app["result"]["summary"]["compared_pair_count"] == 3


def test_compare_preserves_duplicate_row_multiplicity(tmp_path: Path):
    left = tmp_path / "left.csv"
    right = tmp_path / "right.csv"
    left.write_text("x\n1\n1\n", encoding="utf-8")
    right.write_text("x\n1\n", encoding="utf-8")
    result = data.compare_files(left, right)
    assert result["row_comparison"]["shared_row_instances"] == 1
    assert result["row_comparison"]["left_overlap_fraction"] == 0.5
    assert result["row_comparison"]["right_overlap_fraction"] == 1.0
