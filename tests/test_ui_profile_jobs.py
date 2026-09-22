from __future__ import annotations

import time
from pathlib import Path

from ml_lab.ui.jobs import ProfileJobManager


def test_profile_job_reports_status_and_persists_results(tmp_path: Path):
    source = tmp_path / "data.csv"
    source.write_text("x,y\n1,2\n2,4\n3,6\n", encoding="utf-8")
    manager = ProfileJobManager(output_root=tmp_path / "runs")
    job_id = manager.start({"paths": [str(source)], "relationship_rows": 0, "max_rows": 0})

    deadline = time.time() + 10
    job = manager.get(job_id)
    while job and job["status"] not in {"completed", "failed"} and time.time() < deadline:
        time.sleep(0.02)
        job = manager.get(job_id)

    assert job is not None
    assert job["status"] == "completed", job.get("error")
    assert job["stage"] == "Profile complete"
    assert job["artifact_manifest"]["profiles"]
    assert Path(job["artifact_manifest"]["manifest_path"]).is_file()
