"""Server-rendered typed controls for the unified experiment model workspace.

Only the stable Sprint 3.1 registry supplies field definitions. This module
imports neither Flask nor model-training engines, and does not execute tasks.
"""
from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any


class ModelFormError(ValueError):
    """Malformed form input (before scientific configuration validation)."""


def _fields(schema: Mapping[str, Any], prefix: str = ""):
    for group, fields in schema["controls"].items():
        for field in fields:
            path = f"{prefix}{group}.{field['name']}"
            if field["type"] == "object":
                yield from _nested(field, path)
            else:
                if field["name"] == "checkpoint_path" and group == "training_config":
                    continue  # Allocated only by the future training-run manager.
                yield path, field


def _nested(field: Mapping[str, Any], path: str):
    for child in field.get("fields", []):
        name = f"{path}.{child['name']}"
        if child["type"] == "object":
            yield from _nested(child, name)
        else:
            yield name, child


def visible_controls(schema: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Return a copy with non-editable checkpoint paths removed at any depth."""
    def without_reserved(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for raw in items:
            field = dict(raw)
            if field["name"] == "checkpoint_path":
                continue
            if field["type"] == "object":
                field["fields"] = without_reserved(field.get("fields", []))
            result.append(field)
        return result

    return {group: without_reserved(fields) for group, fields in schema["controls"].items()}


def _parse_value(raw: str, field: Mapping[str, Any], label: str) -> Any:
    kind = field["type"]
    try:
        if kind == "string":
            return raw
        if kind == "enum":
            if raw not in field.get("choices", []):
                raise ValueError("value not in allowed choices")
            return raw
        if kind == "boolean":
            if raw not in ("true", "false"):
                raise ValueError("expected true or false")
            return raw == "true"
        if kind == "integer":
            # int('1.0') is invalid, and int('True') must not be silently accepted.
            if not raw.lstrip("+-").isdigit():
                raise ValueError("expected an integer")
            return int(raw)
        if kind == "number":
            value = float(raw)
            if not math.isfinite(value):
                raise ValueError("expected a finite number")
            return value
        if kind == "union":
            # For the current stable registry (e.g. PCA n_components) this
            # encodes a JSON scalar such as 3, 0.95, or null.
            value = json.loads(raw)
            if isinstance(value, (list, dict)):
                raise ValueError("expected a scalar JSON value")
            return value
        if kind == "array":
            if raw.lstrip().startswith("["):
                items = json.loads(raw)
                if not isinstance(items, list):
                    raise ValueError("expected a JSON array")
            else:
                items = [part.strip() for part in raw.split(",")]
                if not all(items):
                    raise ValueError("array has an empty element")
            item_type = field.get("items", {"type": "string"})
            result = []
            for index, item in enumerate(items):
                if isinstance(item, (list, dict)):
                    raise ValueError("nested arrays/objects are not supported in the compact editor")
                # Prefer preserving JSON value types when explicitly supplied.
                if isinstance(item, (int, float, bool)):
                    result.append(item)
                elif item_type["type"] == "string":
                    result.append(str(item))
                else:
                    result.append(_parse_value(str(item), item_type, f"{label}[{index}]"))
            return result
    except (ValueError, TypeError, json.JSONDecodeError) as exc:
        raise ModelFormError(f"{label}: {exc}") from exc
    raise ModelFormError(f"{label}: unsupported field type {kind!r}")


def read_model_form(form: Any, schema: Mapping[str, Any]) -> dict[str, Any]:
    """Read only advertised, typed controls, ignoring untouched default fields.

    A blank field delegates its default to the authoritative scientific config.
    Multi-select choices are checked here and again by the model registry.
    """
    leaves = dict(_fields(schema))
    supplied = {name.removeprefix("ctl.") for name in form if name.startswith("ctl.")}
    unknown = supplied - set(leaves)
    if unknown:
        raise ModelFormError(f"Unknown model control: {sorted(unknown)[0]}")
    result: dict[str, Any] = {}
    for path, field in leaves.items():
        key = f"ctl.{path}"
        if field["type"] == "multi_select":
            chosen = form.getlist(key) if hasattr(form, "getlist") else form.get(key, [])
            if isinstance(chosen, str):
                chosen = [chosen]
            if chosen:
                allowed = set(field["choices"])
                if len(chosen) != len(set(chosen)) or any(choice not in allowed for choice in chosen):
                    raise ModelFormError(f"{path}: unknown or duplicate selection")
                value = list(chosen)
            else:
                continue
        else:
            raw = form.get(key, "")
            if not isinstance(raw, str):
                raise ModelFormError(f"{path}: expected a single control value")
            raw = raw.strip()
            if raw == "":
                continue
            value = _parse_value(raw, field, path)
        target = result
        parts = path.split(".")
        if parts[0] == "task_options":
            parts.pop(0)  # Registry emits these as top-level task settings.
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return result


def model_form_values(settings: Mapping[str, Any] | None) -> dict[str, Any]:
    """Flatten persisted explicit selections for round-trip editing."""
    result: dict[str, Any] = {}

    def walk(node: Mapping[str, Any], prefix: str = "") -> None:
        for name, value in node.items():
            key = f"{prefix}{name}"
            if isinstance(value, Mapping):
                walk(value, key + ".")
            elif isinstance(value, bool):
                result[key] = "true" if value else "false"
            elif isinstance(value, list):
                result[key] = value  # checkboxes or comma-separated array
            elif value is not None:
                result[key] = str(value)
    if settings:
        walk(settings)
    for key in ("test_size", "estimators", "hidden_dim"):
        if key in result:
            result[f"task_options.{key}"] = result.pop(key)
    return result


def submitted_form_values(form: Any) -> dict[str, Any]:
    """Keep a failed submission visible without trusting it as configuration."""
    result: dict[str, Any] = {}
    for key in form:
        if key.startswith("ctl."):
            vals = form.getlist(key) if hasattr(form, "getlist") else [form[key]]
            result[key[4:]] = vals if len(vals) > 1 else (vals[0] if vals else "")
    return result


def control_placeholder(field: Mapping[str, Any]) -> str:
    """Engine defaults are hints; blank fields remain *unset* in the revision."""
    if field.get("default_nonfinite"):
        return "Engine-defined default"
    value = field.get("default")
    if value is None:
        return "Optional — engine default"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, tuple)):
        return ", ".join(str(item) for item in value)
    return str(value)


__all__ = [
    "ModelFormError", "visible_controls", "read_model_form", "model_form_values",
    "submitted_form_values", "control_placeholder",
]
