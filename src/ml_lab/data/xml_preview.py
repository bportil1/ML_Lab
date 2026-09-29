from __future__ import annotations

import hashlib
import json
import math
import xml.etree.ElementTree as ET
from collections import OrderedDict
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .types import (
    XmlCollectionPlan,
    XmlFieldCandidate,
    XmlPreviewColumn,
    XmlPreviewTable,
    XmlStructureArtifact,
    XmlTabularPreview,
)
from .xml_ingest import inspect_xml_structure
from .xml_rules import build_xml_collection_plan
from .xml_selection import build_xml_record_selection


class XmlPreviewError(ValueError):
    """Raised when a configured XML extraction cannot be previewed safely."""


def _decode_pointer_segment(value: str) -> str:
    return value.replace("~1", "/").replace("~0", "~")


def _canonical_parts(path: str) -> tuple[str, ...]:
    if not path.startswith("/"):
        raise XmlPreviewError(f"invalid canonical XML path: {path}")
    return tuple(_decode_pointer_segment(part) for part in path.split("/")[1:] if part != "")


def _field_attribute_name(field: XmlFieldCandidate) -> str:
    if field.namespace_uri:
        return f"{{{field.namespace_uri}}}{field.local_name}"
    return field.local_name


def _relative_element_parts(field: XmlFieldCandidate, record_root: str) -> tuple[str, ...]:
    field_parts = _canonical_parts(field.source_element_canonical_path)
    root_parts = _canonical_parts(record_root)
    if field_parts[: len(root_parts)] != root_parts:
        raise XmlPreviewError(f"field is outside the configured record root: {field.relative_path}")
    return field_parts[len(root_parts) :]


def _descendants_at_path(root: ET.Element, relative_parts: tuple[str, ...]) -> list[ET.Element]:
    current = [root]
    for tag in relative_parts:
        next_level: list[ET.Element] = []
        for parent in current:
            next_level.extend(child for child in list(parent) if str(child.tag) == tag)
        current = next_level
        if not current:
            break
    return current


def _scalar_from_element(element: ET.Element, field: XmlFieldCandidate) -> str | None:
    if field.kind == "attribute":
        value = element.attrib.get(_field_attribute_name(field))
        return value if value is not None else None
    text = element.text.strip() if element.text and element.text.strip() else None
    return text


def _field_values(record: ET.Element, field: XmlFieldCandidate, record_root: str) -> list[str | None]:
    elements = _descendants_at_path(record, _relative_element_parts(field, record_root))
    return [_scalar_from_element(element, field) for element in elements]


def _field_values_within_branch(
    branch_element: ET.Element,
    field: XmlFieldCandidate,
    branch_canonical_path: str,
) -> list[str | None]:
    field_parts = _canonical_parts(field.source_element_canonical_path)
    branch_parts = _canonical_parts(branch_canonical_path)
    if field_parts[: len(branch_parts)] != branch_parts:
        return []
    elements = _descendants_at_path(branch_element, field_parts[len(branch_parts) :])
    return [_scalar_from_element(element, field) for element in elements]


def _branch_elements(record: ET.Element, branch_canonical_path: str, record_root: str) -> list[ET.Element]:
    branch_parts = _canonical_parts(branch_canonical_path)
    root_parts = _canonical_parts(record_root)
    if branch_parts[: len(root_parts)] != root_parts:
        return []
    return _descendants_at_path(record, branch_parts[len(root_parts) :])


def _first_non_null(values: Iterable[str | None]) -> str | None:
    for value in values:
        if value is not None:
            return value
    return None


def _last_non_null(values: Iterable[str | None]) -> str | None:
    result = None
    for value in values:
        if value is not None:
            result = value
    return result


def _number(value: str | None) -> float | None:
    if value is None or value.strip() == "":
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _aggregate(values: Iterable[str | None], operation: str) -> tuple[int | float | None, int]:
    numeric: list[float] = []
    invalid = 0
    for value in values:
        if value is None or value.strip() == "":
            continue
        parsed = _number(value)
        if parsed is None:
            invalid += 1
        else:
            numeric.append(parsed)
    if not numeric:
        return None, invalid
    if operation == "sum":
        result = sum(numeric)
    elif operation == "mean":
        result = sum(numeric) / len(numeric)
    elif operation == "min":
        result = min(numeric)
    elif operation == "max":
        result = max(numeric)
    else:  # Validated by the collection plan.
        raise XmlPreviewError(f"unsupported aggregate operation: {operation}")
    if float(result).is_integer():
        return int(result), invalid
    return round(float(result), 12), invalid


def _unique_name(base: str, used: set[str]) -> str:
    candidate = base or "value"
    if candidate not in used:
        used.add(candidate)
        return candidate
    counter = 2
    while f"{candidate}__{counter}" in used:
        counter += 1
    result = f"{candidate}__{counter}"
    used.add(result)
    return result


def _column(
    name: str,
    fields: Iterable[XmlFieldCandidate],
    *,
    strategy: str,
    branch: str | None = None,
    dynamic: bool = False,
) -> XmlPreviewColumn:
    items = tuple(fields)
    return XmlPreviewColumn(
        name=name,
        source_field_ids=tuple(field.field_id for field in items),
        source_relative_paths=tuple(field.relative_path for field in items),
        source_element_canonical_paths=tuple(dict.fromkeys(field.source_element_canonical_path for field in items)),
        strategy=strategy,
        repeated_branch_canonical_path=branch,
        dynamic=dynamic,
    )


def _rule_by_branch(plan: XmlCollectionPlan) -> dict[str, Any]:
    return {
        branch.canonical_path: branch.rule
        for branch in plan.repeated_branches
        if branch.rule is not None and branch.rule.strategy is not None
    }


def _deepest_branch_for_field(field: XmlFieldCandidate, plan: XmlCollectionPlan) -> str | None:
    matches = [
        branch.canonical_path
        for branch in plan.repeated_branches
        if field.source_element_canonical_path == branch.canonical_path
        or field.source_element_canonical_path.startswith(branch.canonical_path.rstrip("/") + "/")
    ]
    return max(matches, key=lambda item: len(_canonical_parts(item))) if matches else None


def _preview_signature(
    artifact: XmlStructureArtifact,
    record_root: str,
    selected_field_ids: tuple[str, ...],
    rules: Iterable[Mapping[str, Any]] | None,
) -> str:
    payload = {
        "source_fingerprint": artifact.source_fingerprint,
        "record_root_canonical_path": record_root,
        "selected_field_ids": list(selected_field_ids),
        "rules": list(rules or ()),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return "sha256:" + hashlib.sha256(raw).hexdigest()


def preview_xml_tabularization(
    path: str | Path,
    artifact: XmlStructureArtifact,
    record_root_canonical_path: str,
    selected_field_ids: Iterable[str],
    *,
    rules: Iterable[Mapping[str, Any]] | None = None,
    max_rows: int | None = 25,
    max_child_rows: int | None = 100,
    max_output_rows: int | None = 1000,
) -> XmlTabularPreview:
    """Build a bounded, non-destructive XML-to-table preview.

    The preview is intentionally not a registered ML Lab dataset. Sprint 6 materializes
    only enough rows to let the user inspect field-to-column mapping and rule effects.
    """

    if max_rows is not None and (max_rows < 1 or max_rows > 200):
        raise XmlPreviewError("max_rows must be between 1 and 200")
    if max_child_rows is not None and (max_child_rows < 1 or max_child_rows > 1000):
        raise XmlPreviewError("max_child_rows must be between 1 and 1000")
    if max_output_rows is not None and (max_output_rows < 1 or max_output_rows > 10_000):
        raise XmlPreviewError("max_output_rows must be between 1 and 10000")

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"not a file: {source}")

    # Re-run the bounded safe structural parser so direct API callers cannot bypass the
    # DTD/entity policy and so a stale artifact is detected before reading values.
    current = inspect_xml_structure(source)
    if current.source_fingerprint != artifact.source_fingerprint:
        raise XmlPreviewError("XML source changed after structural analysis; reload the structure before previewing")

    selection = build_xml_record_selection(artifact, record_root_canonical_path)
    selected_tuple = tuple(dict.fromkeys(str(field_id) for field_id in selected_field_ids))
    if not selected_tuple:
        raise XmlPreviewError("select at least one XML field before building a table preview")
    plan = build_xml_collection_plan(artifact, record_root_canonical_path, selected_tuple, rules=rules)
    if not plan.ready_for_preview:
        raise XmlPreviewError(
            "repeated-branch rules are not ready for preview "
            f"({plan.unresolved_branch_count} unresolved, {plan.invalid_rule_count} invalid)"
        )

    field_by_id = {field.field_id: field for field in selection.fields}
    selected_fields = tuple(field_by_id[field_id] for field_id in selected_tuple)
    branch_by_path = {branch.canonical_path: branch for branch in plan.repeated_branches}
    rule_by_path = _rule_by_branch(plan)
    deepest_branch = {field.field_id: _deepest_branch_for_field(field, plan) for field in selected_fields}
    nested_branch_paths = {
        branch.canonical_path
        for branch in plan.repeated_branches
        if branch.parent_repeated_branch_canonical_path is not None
    }

    warnings = list(plan.warnings)
    if nested_branch_paths:
        warnings.append(
            "Nested repeated branches are previewed with the outermost repeated branch as the row-shaping boundary. "
            "Nested rules remain in the contract, but descendant values are reduced inside each outer occurrence in this preview."
        )

    used_names: set[str] = set()
    columns: list[XmlPreviewColumn] = []
    field_column_name: dict[str, str] = {}

    direct_fields = [field for field in selected_fields if deepest_branch[field.field_id] is None]
    for field in direct_fields:
        name = _unique_name(field.relative_path, used_names)
        field_column_name[field.field_id] = name
        columns.append(_column(name, [field], strategy="direct"))

    # Pre-create stable columns for strategies whose output shape does not depend on values.
    for branch in plan.repeated_branches:
        if branch.parent_repeated_branch_canonical_path is not None:
            continue
        rule = rule_by_path.get(branch.canonical_path)
        if rule is None:
            continue
        branch_fields = [field_by_id[field_id] for field_id in branch.selected_field_ids]
        strategy = rule.strategy
        if strategy in {"first", "last", "join", "aggregate", "explode_rows"}:
            for field in branch_fields:
                name = _unique_name(field.relative_path, used_names)
                field_column_name[field.field_id] = name
                columns.append(_column(name, [field], strategy=strategy, branch=branch.canonical_path))
        elif strategy == "count":
            name = _unique_name(f"{branch.relative_path}.__count", used_names)
            columns.append(_column(name, branch_fields, strategy="count", branch=branch.canonical_path))
        elif strategy == "keep_nested":
            name = _unique_name(branch.relative_path, used_names)
            columns.append(_column(name, branch_fields, strategy="keep_nested", branch=branch.canonical_path))
        # pivot columns are value-dependent; separate_table columns live in a child preview.

    record_root_parts = _canonical_parts(record_root_canonical_path)
    raw_records: list[ET.Element] = []
    stack: list[str] = []
    with source.open("rb") as handle:
        for event, element in ET.iterparse(handle, events=("start", "end")):
            if event == "start":
                stack.append(str(element.tag))
                continue
            if tuple(stack) == record_root_parts and (max_rows is None or len(raw_records) < max_rows):
                # Detach by serializing this bounded subtree. The parser can then clear the original.
                raw_records.append(ET.fromstring(ET.tostring(element, encoding="utf-8")))
            if tuple(stack) == record_root_parts:
                element.clear()
            stack.pop()
            if max_rows is not None and len(raw_records) >= max_rows:
                # The structural artifact already contains the total record-root occurrence count.
                # Stop reading values once the requested preview is full.
                break

    main_rows: list[dict[str, Any]] = []
    child_rows: OrderedDict[str, list[dict[str, Any]]] = OrderedDict()
    child_columns: dict[str, list[XmlPreviewColumn]] = {}
    aggregate_invalid_counts: dict[str, int] = {}
    pivot_column_names: dict[tuple[str, str], str] = {}
    output_truncated = False

    top_branches = [branch for branch in plan.repeated_branches if branch.parent_repeated_branch_canonical_path is None]

    for record_index, record in enumerate(raw_records, start=1):
        base: dict[str, Any] = {}
        for field in direct_fields:
            values = _field_values(record, field, record_root_canonical_path)
            base[field_column_name[field.field_id]] = _first_non_null(values)
        row_variants = [base]

        for branch in top_branches:
            rule = rule_by_path.get(branch.canonical_path)
            if rule is None or rule.strategy is None:
                continue
            branch_elements = _branch_elements(record, branch.canonical_path, record_root_canonical_path)
            branch_fields = [field_by_id[field_id] for field_id in branch.selected_field_ids]
            strategy = rule.strategy

            if strategy == "explode_rows":
                expanded: list[dict[str, Any]] = []
                occurrences = branch_elements or [None]
                for row in row_variants:
                    for branch_element in occurrences:
                        if max_output_rows is not None and len(expanded) >= max_output_rows:
                            output_truncated = True
                            break
                        next_row = dict(row)
                        for field in branch_fields:
                            values = [] if branch_element is None else _field_values_within_branch(branch_element, field, branch.canonical_path)
                            next_row[field_column_name[field.field_id]] = _first_non_null(values)
                        expanded.append(next_row)
                    if output_truncated:
                        break
                row_variants = expanded
                continue

            if strategy == "separate_table":
                table_name = rule.separate_table_name or branch.relative_path.replace("/", "_") or "child"
                rows = child_rows.setdefault(table_name, [])
                if table_name not in child_columns:
                    parent_col = XmlPreviewColumn(name="__parent_record", strategy="link")
                    child_columns[table_name] = [parent_col]
                    local_used = {"__parent_record"}
                    for field in branch_fields:
                        name = _unique_name(field.relative_path.removeprefix(branch.relative_path + "/") or field.name, local_used)
                        child_columns[table_name].append(_column(name, [field], strategy="separate_table", branch=branch.canonical_path))
                if max_child_rows is None or len(rows) < max_child_rows:
                    field_names = {
                        column.source_field_ids[0]: column.name
                        for column in child_columns[table_name]
                        if column.source_field_ids
                    }
                    for branch_element in branch_elements:
                        if max_child_rows is not None and len(rows) >= max_child_rows:
                            break
                        child = {"__parent_record": record_index}
                        for field in branch_fields:
                            values = _field_values_within_branch(branch_element, field, branch.canonical_path)
                            child[field_names[field.field_id]] = _first_non_null(values) if len(values) <= 1 else values
                        rows.append(child)
                continue

            if strategy == "count":
                count_column = next(
                    column.name
                    for column in columns
                    if column.strategy == "count" and column.repeated_branch_canonical_path == branch.canonical_path
                )
                for row in row_variants:
                    row[count_column] = len(branch_elements)
                continue

            if strategy == "keep_nested":
                nested_column = next(
                    column.name
                    for column in columns
                    if column.strategy == "keep_nested" and column.repeated_branch_canonical_path == branch.canonical_path
                )
                nested_values: list[dict[str, Any]] = []
                for branch_element in branch_elements:
                    item: dict[str, Any] = {}
                    for field in branch_fields:
                        relative = field.relative_path.removeprefix(branch.relative_path + "/") or field.name
                        values = _field_values_within_branch(branch_element, field, branch.canonical_path)
                        item[relative] = values[0] if len(values) <= 1 else values
                    nested_values.append(item)
                for row in row_variants:
                    row[nested_column] = nested_values
                continue

            if strategy == "pivot":
                key_field = field_by_id.get(rule.pivot_key_field_id or "")
                value_field = field_by_id.get(rule.pivot_value_field_id or "")
                if key_field is None or value_field is None:
                    raise XmlPreviewError("pivot rule references fields that are no longer selected")
                pivot_pairs: list[tuple[str, Any]] = []
                for branch_element in branch_elements:
                    key = _first_non_null(_field_values_within_branch(branch_element, key_field, branch.canonical_path))
                    value = _first_non_null(_field_values_within_branch(branch_element, value_field, branch.canonical_path))
                    if key is not None:
                        pivot_pairs.append((key, value))
                for key, value in pivot_pairs:
                    pivot_key = (branch.canonical_path, key)
                    name = pivot_column_names.get(pivot_key)
                    if name is None:
                        wanted = f"{branch.relative_path}[{key}]"
                        name = _unique_name(wanted, used_names)
                        pivot_column_names[pivot_key] = name
                        columns.append(
                            _column(
                                name,
                                [key_field, value_field],
                                strategy="pivot",
                                branch=branch.canonical_path,
                                dynamic=True,
                            )
                        )
                    for row in row_variants:
                        if name in row and row[name] != value:
                            warnings.append(
                                f"Pivot key {key!r} occurred more than once in one record; the last preview value was retained."
                            )
                        row[name] = value
                continue

            for field in branch_fields:
                all_values: list[str | None] = []
                for branch_element in branch_elements:
                    all_values.extend(_field_values_within_branch(branch_element, field, branch.canonical_path))
                if strategy == "first":
                    value: Any = _first_non_null(all_values)
                elif strategy == "last":
                    value = _last_non_null(all_values)
                elif strategy == "join":
                    value = (rule.join_delimiter or "").join(str(item) for item in all_values if item is not None)
                elif strategy == "aggregate":
                    value, invalid = _aggregate(all_values, rule.aggregate_operation or "sum")
                    if invalid:
                        aggregate_invalid_counts[field.relative_path] = aggregate_invalid_counts.get(field.relative_path, 0) + invalid
                else:
                    raise XmlPreviewError(f"unsupported preview strategy: {strategy}")
                for row in row_variants:
                    row[field_column_name[field.field_id]] = value

        if max_output_rows is None:
            main_rows.extend(row_variants)
        else:
            remaining = max_output_rows - len(main_rows)
            if len(row_variants) > remaining:
                main_rows.extend(row_variants[:remaining])
                output_truncated = True
            else:
                main_rows.extend(row_variants)
            if len(main_rows) >= max_output_rows and record_index < len(raw_records):
                output_truncated = True
                break

    for relative_path, invalid_count in sorted(aggregate_invalid_counts.items()):
        warnings.append(
            f"Aggregate preview ignored {invalid_count} non-numeric value(s) for {relative_path}."
        )

    # Ensure every preview row exposes every discovered column (including dynamic pivot columns).
    for row in main_rows:
        for column in columns:
            row.setdefault(column.name, None)

    root_profile = next(
        (profile for profile in artifact.element_profiles if profile.canonical_path == record_root_canonical_path),
        None,
    )
    estimated_records = root_profile.occurrence_count if root_profile is not None else len(raw_records)
    root_truncated = estimated_records > len(raw_records)
    truncated = root_truncated or output_truncated
    if root_truncated:
        warnings.append(f"Main preview is limited to the first {len(raw_records)} record-root occurrence(s).")
    if output_truncated and max_output_rows is not None:
        warnings.append(f"Expanded main-table preview is limited to {max_output_rows} output row(s).")

    child_tables: list[XmlPreviewTable] = []
    for name, rows in child_rows.items():
        was_truncated = max_child_rows is not None and len(rows) >= max_child_rows
        if was_truncated:
            warnings.append(f"Child-table preview {name!r} is limited to {max_child_rows} row(s).")
        child_tables.append(
            XmlPreviewTable(
                name=name,
                role="child",
                columns=tuple(child_columns[name]),
                rows=tuple(rows),
                preview_row_count=len(rows),
                estimated_total_rows=None,
                truncated=was_truncated,
            )
        )

    signature = _preview_signature(artifact, record_root_canonical_path, selected_tuple, rules)
    return XmlTabularPreview(
        source_fingerprint=artifact.source_fingerprint,
        preview_signature=signature,
        record_root_path=selection.record_root_path,
        record_root_canonical_path=selection.record_root_canonical_path,
        selected_field_ids=selected_tuple,
        main_table=XmlPreviewTable(
            name="main",
            role="main",
            columns=tuple(columns),
            rows=tuple(main_rows),
            preview_row_count=len(main_rows),
            estimated_total_rows=(None if any(rule_by_path.get(branch.canonical_path) and rule_by_path[branch.canonical_path].strategy == "explode_rows" for branch in top_branches) else estimated_records),
            truncated=truncated,
        ),
        child_tables=tuple(child_tables),
        warnings=tuple(dict.fromkeys(warnings)),
    )
