from __future__ import annotations

from pathlib import Path

from ml_lab import data


def test_profile_run_persists_collection_individual_profiles_and_can_be_reopened(tmp_path: Path):
    source = tmp_path / "data.csv"
    source.write_text("x,y\n1,2\n2,4\n3,6\n", encoding="utf-8")
    collection = data.profile_paths([source], relationship_rows=0)

    manifest = data.persist_profile_run(collection, output_root=tmp_path / "out", run_id="test-run")
    assert Path(manifest["collection_path"]).is_file()
    assert Path(manifest["profiles"][0]["profile_path"]).is_file()

    reopened = data.load_profile_run(output_root=tmp_path / "out", run_id="test-run")
    profile = data.load_persisted_profile(reopened, 0)
    assert profile["schema"] == "ml-lab.data-profile@1"
    assert profile["relative_path"] == "data.csv"
    recent = data.recent_profile_runs(output_root=tmp_path / "out")
    assert recent[0]["run_id"] == "test-run"
