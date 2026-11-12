#!/usr/bin/env python3
"""Read-only, stdlib-only inventory of ML_Lab's UI and storage surfaces.

Use this before each Unified Workbench migration increment. It does not import
ML_Lab, instantiate a Flask app, read datasets, or modify project state.
"""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def _literal_route(decorator: ast.expr) -> tuple[str, str] | None:
    if not isinstance(decorator, ast.Call) or not isinstance(decorator.func, ast.Attribute):
        return None
    if not isinstance(decorator.func.value, ast.Name):
        return None
    if decorator.func.value.id != "blueprint":
        return None
    if decorator.func.attr not in {"get", "post", "put", "delete", "route"}:
        return None
    if not decorator.args or not isinstance(decorator.args[0], ast.Constant):
        return None
    path = decorator.args[0].value
    if not isinstance(path, str):
        return None
    method = decorator.func.attr.upper()
    if method == "ROUTE":
        method = "GET"
        for keyword in decorator.keywords:
            if keyword.arg == "methods" and isinstance(keyword.value, (ast.List, ast.Tuple)):
                values = [element.value for element in keyword.value.elts if isinstance(element, ast.Constant)]
                if values and all(isinstance(value, str) for value in values):
                    method = ",".join(sorted(values))
    return path, method


def snapshot(root: Path = ROOT) -> dict[str, object]:
    project = (root / "pyproject.toml").read_text(encoding="utf-8")
    version_match = re.search(r'^version\s*=\s*"([^"]+)"', project, re.M)
    routes: list[dict[str, str]] = []
    for source in sorted((root / "src/ml_lab/ui").glob("*.py")):
        tree = ast.parse(source.read_text(encoding="utf-8"), filename=str(source))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            for decorator in node.decorator_list:
                route = _literal_route(decorator)
                if route is not None:
                    path, methods = route
                    routes.append({
                        "path": path, "methods": methods,
                        "endpoint": node.name,
                        "source": source.relative_to(root).as_posix(),
                    })
    routes.sort(key=lambda record: (record["path"], record["methods"], record["endpoint"]))
    base_path = root / "src/ml_lab/ui/templates/ml_lab_ui/base.html"
    base = base_path.read_text(encoding="utf-8") if base_path.exists() else ""
    navigation = re.findall(r'>\s*(Overview|Workspace|Data|Experiments|Compare|Transform|Provenance)\s*</a>', base)
    components = {
        "core_profile": (root / "src/ml_lab/data/profiling.py").is_file(),
        "xml_collection_inventory": (root / "src/ml_lab/data/xml_collection.py").is_file(),
        "xml_collection_recipe": (root / "src/ml_lab/data/xml_collection_recipe.py").is_file(),
        "xml_collection_export": (root / "src/ml_lab/data/xml_collection_export.py").is_file(),
        "experiment_sqlite_store": (root / "src/ml_lab/experiments/store.py").is_file(),
        "experiment_dataset_selection": (root / "src/ml_lab/experiments/datasets.py").is_file(),
        "pca_representation": (root / "src/ml_lab/representation/pca.py").is_file(),
        "mlp_autoencoder": (root / "src/ml_lab/representation/training.py").is_file(),
        "transformer_autoencoder": (root / "src/ml_lab/representation/transformer_training.py").is_file(),
    }
    return {
        "audit_schema": "ml-lab.unified-workbench-source-inventory@1",
        "project_version": version_match.group(1) if version_match else None,
        "ui_routes": routes,
        "xml_collection_route_paths": sorted({r["path"] for r in routes if r["path"].startswith("/data/xml/collection")}),
        "legacy_header_navigation": navigation,
        "component_presence": components,
        "source_file_count": len(list((root / "src/ml_lab").rglob("*.py"))),
    }


def main() -> None:
    print(json.dumps(snapshot(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
