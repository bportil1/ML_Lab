from __future__ import annotations

import subprocess
import sys


def test_stable_transformer_symbols_do_not_eagerly_import_torch():
    code = """
import sys
from ml_lab import representation
assert 'torch' not in sys.modules
assert representation.TransformerAutoencoderConfig is not None
assert 'torch' not in sys.modules
"""
    completed = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
