from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from .types import (
    XmlCollectionPlan,
    XmlCollectionRule,
    XmlFieldCandidate,
    XmlRepeatedBranch,
    XmlStructureArtifact,
)
from .xml_selection import build_xml_record_selection


class XmlCollectionRuleError(ValueError):
    """Raised when XML one-to-many rule configuration cannot be associated with the selection."""


_STRATEGY_CATALOG: tuple[dict[str, Any], ...] = (
    {
        "id": "keep_nested",
        "label": "Keep nested",
        "description": "Preserve the repeated branch as a nested value for one parent record.",
        "options": (),
    },
    {
        "id": "first",
        "label": "First",
        "description": "Use the first repeated occurrence for selected descendant fields.",
        "options": (),
    },
    {
        "id": "last",
        "label": "Last",
        "description": "Use the last repeated occurrence for selected descendant fields.",
        "options": (),
    },
    {
        "id": "count",
        "label": "Count",
        "description": "Replace descendant values with the number of repeated occurrences.",
        "options": (),
    },
    {
        "id": "join",
        "label": "Join",
        "description": "Join repeated scalar values into one cell using an explicit delimiter.",
        "options": ("join_delimiter",),
    },
    {
        "id": "aggregate",
        "label": "Aggregate",
        "description": "Aggregate repeated scalar values using an explicit numeric operation.",
        "options": ("aggregate_operation",),
    },
    {
        "id": "pivot",
        "label": "Pivot",
        "description": "Use one selected field as keys and another selected field as values.",
        "options": ("pivot_key_field_id", "pivot_value_field_id"),
    },
    {
        "id": "explode_rows",
        "label": "Explode rows",
        "description": "Emit one output row per repeated occurrence while retaining parent fields.",
        "options": (),
    },
    {
        "id": "separate_table",
        "label": "Separate table",
        "description": "Materialize the repeated branch as a linked child table.",
        "options": ("separate_table_name",),
    },
)
_STRATEGY_IDS = {item["id"] for item in _STRATEGY_CATALOG}
_AGGREGATE_OPERATIONS = {"sum", "mean", "min", "max"}


def xml_collection_strategy_catalog() -> tuple[dict[str, Any], ...]:
    """Return immutable-by-convention metadata for XML repeated-branch strategies."""

    return tuple({**item, "options": tuple(item["options"])} for item in _STRATEGY_CATALOG)


def _is_descendant_or_self(path: str, root: str) -> bool:
    return path == root or path.startswith(root.rstrip("/") + "/")


def _relative_display_path(path: str, root_path: str) -> str:
    if path == root_path:
        return "."
    prefix = root_path.rstrip("/") + "/"
    return path[len(prefix):] if path.startswith(prefix) else path


def _rule_input_map(rules: Iterable[Mapping[str, Any]] | None) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for raw in rules or ():
        branch = str(raw.get("branch_canonical_path") or "").strip()
        if not branch:
            raise XmlCollectionRuleError("each XML collection rule requires branch_canonical_path")
        if branch in result:
            raise XmlCollectionRuleError(f"duplicate XML collection rule for branch: {branch}")
        result[branch] = raw
    return result


def _clean_optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value)
    return text if text else None


def _normalize_rule(
    branch_path: str,
    raw: Mapping[str, Any] | None,
    branch_fields: tuple[XmlFieldCandidate, ...],
) -> XmlCollectionRule:
    if raw is None:
        return XmlCollectionRule(
            branch_canonical_path=branch_path,
            strategy=None,
            valid=False,
            errors=("Choose a transformation strategy for this repeated branch.",),
        )

    strategy = str(raw.get("strategy") or "").strip() or None
    options = raw.get("options")
    if not isinstance(options, Mapping):
        options = raw

    join_delimiter = _clean_optional_text(options.get("join_delimiter"))
    aggregate_operation = _clean_optional_text(options.get("aggregate_operation"))
    pivot_key_field_id = _clean_optional_text(options.get("pivot_key_field_id"))
    pivot_value_field_id = _clean_optional_text(options.get("pivot_value_field_id"))
    separate_table_name = _clean_optional_text(options.get("separate_table_name"))
    errors: list[str] = []

    if strategy not in _STRATEGY_IDS:
        if strategy is None:
            errors.append("Choose a transformation strategy for this repeated branch.")
        else:
            errors.append(f"Unsupported transformation strategy: {strategy}")
    elif strategy == "join":
        if join_delimiter is None:
            errors.append("Join requires an explicit delimiter.")
        elif len(join_delimiter) > 32:
            errors.append("Join delimiter must be 32 characters or fewer.")
    elif strategy == "aggregate":
        if aggregate_operation not in _AGGREGATE_OPERATIONS:
            errors.append("Aggregate requires one of: sum, mean, min, max.")
    elif strategy == "pivot":
        available = {field.field_id for field in branch_fields}
        if not pivot_key_field_id or pivot_key_field_id not in available:
            errors.append("Pivot requires a key field selected from this repeated branch.")
        if not pivot_value_field_id or pivot_value_field_id not in available:
            errors.append("Pivot requires a value field selected from this repeated branch.")
        if pivot_key_field_id and pivot_value_field_id and pivot_key_field_id == pivot_value_field_id:
            errors.append("Pivot key and value fields must be different.")
    elif strategy == "separate_table":
        if separate_table_name is not None and len(separate_table_name) > 128:
            errors.append("Separate-table name must be 128 characters or fewer.")

    return XmlCollectionRule(
        branch_canonical_path=branch_path,
        strategy=strategy,
        join_delimiter=join_delimiter,
        aggregate_operation=aggregate_operation,
        pivot_key_field_id=pivot_key_field_id,
        pivot_value_field_id=pivot_value_field_id,
        separate_table_name=separate_table_name,
        valid=not errors,
        errors=tuple(errors),
    )


def build_xml_collection_plan(
    artifact: XmlStructureArtifact,
    record_root_canonical_path: str,
    selected_field_ids: Iterable[str],
    *,
    rules: Iterable[Mapping[str, Any]] | None = None,
) -> XmlCollectionPlan:
    """Build/validate one-to-many rules for an XML record selection.

    This remains configuration-only. It identifies repeated element boundaries touched by the
    selected scalar fields and validates an explicit strategy for each boundary, but it does not
    read XML values or materialize a table.
    """

    selection = build_xml_record_selection(artifact, record_root_canonical_path)
    field_by_id = {field.field_id: field for field in selection.fields}
    requested_ids: list[str] = []
    seen_ids: set[str] = set()
    for raw_id in selected_field_ids:
        field_id = str(raw_id)
        if field_id in seen_ids:
            continue
        if field_id not in field_by_id:
            raise XmlCollectionRuleError(f"selected field is not available under this record root: {field_id}")
        seen_ids.add(field_id)
        requested_ids.append(field_id)

    selected_fields = tuple(field_by_id[field_id] for field_id in requested_ids)
    profile_by_path = {profile.canonical_path: profile for profile in artifact.element_profiles}
    root_profile = profile_by_path.get(selection.record_root_canonical_path)
    if root_profile is None:  # Defensive; selection builder already checks this.
        raise XmlCollectionRuleError("record root is not present in the XML structure")

    repeated_profiles = [
        profile
        for profile in artifact.element_profiles
        if profile.canonical_path != selection.record_root_canonical_path
        and profile.repeated
        and _is_descendant_or_self(profile.canonical_path, selection.record_root_canonical_path)
        and any(_is_descendant_or_self(field.source_element_canonical_path, profile.canonical_path) for field in selected_fields)
    ]
    repeated_profiles.sort(key=lambda item: (item.depth, item.path))
    repeated_paths = {profile.canonical_path for profile in repeated_profiles}
    raw_rules = _rule_input_map(rules)
    unknown_rule_branches = sorted(set(raw_rules) - repeated_paths)
    if unknown_rule_branches:
        raise XmlCollectionRuleError(
            "collection rule does not match a repeated branch used by the selected fields: "
            + ", ".join(unknown_rule_branches)
        )

    branches: list[XmlRepeatedBranch] = []
    for profile in repeated_profiles:
        branch_fields = tuple(
            field for field in selected_fields
            if _is_descendant_or_self(field.source_element_canonical_path, profile.canonical_path)
        )
        parent_repeat = None
        cursor = profile.canonical_path.rsplit("/", 1)[0]
        while cursor and _is_descendant_or_self(cursor, selection.record_root_canonical_path):
            if cursor in repeated_paths:
                parent_repeat = cursor
                break
            if cursor == selection.record_root_canonical_path:
                break
            if "/" not in cursor[1:]:
                break
            cursor = cursor.rsplit("/", 1)[0]

        rule = _normalize_rule(profile.canonical_path, raw_rules.get(profile.canonical_path), branch_fields)
        branches.append(
            XmlRepeatedBranch(
                path=profile.path,
                canonical_path=profile.canonical_path,
                relative_path=_relative_display_path(profile.path, root_profile.path),
                depth=profile.depth,
                occurrence_count=profile.occurrence_count,
                parent_occurrence_count=profile.parent_occurrence_count,
                min_per_parent=profile.min_per_parent,
                max_per_parent=profile.max_per_parent,
                mean_per_parent=profile.mean_per_parent,
                parent_repeated_branch_canonical_path=parent_repeat,
                selected_field_ids=tuple(field.field_id for field in branch_fields),
                selected_fields=branch_fields,
                rule=rule,
            )
        )

    unresolved = sum(1 for branch in branches if branch.rule is None or branch.rule.strategy is None)
    invalid = sum(1 for branch in branches if branch.rule is not None and branch.rule.strategy is not None and not branch.rule.valid)
    warnings: list[str] = []
    if any(branch.rule and branch.rule.strategy == "aggregate" for branch in branches):
        warnings.append("Aggregate numeric compatibility is checked when the XML table preview reads source values.")
    if any(branch.rule and branch.rule.strategy == "keep_nested" for branch in branches):
        warnings.append("Keep nested preserves hierarchy inside a single tabular cell representation; the source XML remains unchanged.")

    return XmlCollectionPlan(
        source_fingerprint=artifact.source_fingerprint,
        record_root_path=selection.record_root_path,
        record_root_canonical_path=selection.record_root_canonical_path,
        selected_field_ids=tuple(requested_ids),
        repeated_branches=tuple(branches),
        strategy_catalog=xml_collection_strategy_catalog(),
        unresolved_branch_count=unresolved,
        invalid_rule_count=invalid,
        ready_for_preview=(unresolved == 0 and invalid == 0),
        warnings=tuple(warnings),
    )
