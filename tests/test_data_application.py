from __future__ import annotations

from pathlib import Path

import pytest

from ml_lab.application import PayloadError, execute_task


def test_data_inspect_application_task_uses_same_core_inventory(tmp_path: Path):
    source = tmp_path / "table.csv"
    source.write_text("id,value\na,1\nb,2\n", encoding="utf-8")

    result = execute_task("data.inspect", {"paths": [str(source)]})
    assert result["task"] == "data.inspect"
    assert result["result"]["schema"] == "ml-lab.data-inventory@1"
    assert result["result"]["files"][0]["row_count"] == 2


def test_data_inspect_application_requires_paths():
    with pytest.raises(PayloadError):
        execute_task("data.inspect", {})


def test_data_profile_application_task_exposes_statistical_profile(tmp_path: Path):
    source = tmp_path / "profile.csv"
    source.write_text("x,y\n1,2\n2,4\n3,6\n", encoding="utf-8")

    result = execute_task("data.profile", {"paths": [str(source)], "relationship_rows": 0})
    assert result["task"] == "data.profile"
    assert result["result"]["schema"] == "ml-lab.data-profile-collection@1"
    profile = result["result"]["profiles"][0]
    assert profile["columns"][0]["inferred_type"] == "integer"
    assert profile["relationships"][0]["pearson"] > 0.99


def test_data_profile_application_requires_paths():
    with pytest.raises(PayloadError):
        execute_task("data.profile", {})
