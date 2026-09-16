from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def _run(code: str) -> None:
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    subprocess.run([sys.executable, "-c", code], check=True, env=env)


def test_import_ml_lab_ui_does_not_load_flask_or_torch():
    _run(
        "import sys; import ml_lab.ui; "
        "assert 'flask' not in sys.modules; "
        "assert 'torch' not in sys.modules"
    )


def test_import_ml_lab_still_does_not_load_ui_or_flask():
    _run(
        "import sys; import ml_lab; "
        "assert 'ml_lab.ui' not in sys.modules; "
        "assert 'flask' not in sys.modules"
    )
