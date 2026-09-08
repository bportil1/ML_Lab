from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys


def test_import_ml_lab_does_not_load_flask_or_experimental():
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    code = (
        "import sys; import ml_lab; "
        "assert 'flask' not in sys.modules; "
        "assert 'ml_lab.experimental' not in sys.modules; assert 'torch' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True, env=env)
