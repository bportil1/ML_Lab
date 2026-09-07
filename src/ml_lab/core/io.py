from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def ensure_output_dir(path: str | Path) -> Path:
    output = Path(path)
    output.mkdir(parents=True, exist_ok=True)
    return output


def write_json(path: str | Path, payload: Any) -> Path:
    target = Path(path)
    target.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return target
