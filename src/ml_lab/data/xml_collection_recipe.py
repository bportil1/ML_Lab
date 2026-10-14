"""XML-C4: structural validation of a shared extraction recipe across reviewed sources.

Read-only: this module does not extract values, write CSV files, or authorize merging.
It delegates every source's record/field/rule semantics to the existing single-file
XML machinery. Structural similarity is not a substitute for this validation.
"""
from __future__ import annotations

import json
import stat
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .xml_collection import XmlCollectionInventory
from .xml_collection_review import review_xml_collection
from .xml_ingest import inspect_xml_structure
from .xml_rules import build_xml_collection_plan
from .xml_selection import build_xml_record_selection


class XmlCollectionRecipeError(ValueError):
    """Raised when a collection recipe request is malformed or sources have changed."""


def _review_sources(inventory: XmlCollectionInventory, similarity: Mapping[str, Any],
                    selected_paths: Iterable[str], acknowledge_variants: bool,
                    *, max_source_bytes: int) -> tuple[dict[str, Any], dict[str, Any]]:
    if similarity.get("root_path") != inventory.root_path:
        raise XmlCollectionRecipeError("similarity report is for a different XML directory")
    review = review_xml_collection(similarity, selected_paths, acknowledge_variants=acknowledge_variants)
    if review["status"] != "ready_for_recipe_design":
        raise XmlCollectionRecipeError("collection review is blocked: " + "; ".join(review["blocking_reasons"]))
    inventory_items = {item.relative_path: item for item in inventory.files}
    similarity_items = {item["relative_path"]: item for item in similarity["sources"]}
    root = Path(inventory.root_path).expanduser().resolve(strict=True)
    artifacts: dict[str, Any] = {}
    for selection in review["selected"]:
        rel = selection["relative_path"]
        item = inventory_items.get(rel)
        evidence = similarity_items.get(rel)
        if item is None or evidence is None or item.status != "parsed" or evidence["status"] != "analyzed":
            raise XmlCollectionRecipeError(f"missing inspected source evidence: {rel}")
        if not item.sha256 or evidence.get("source_fingerprint") != f"sha256:{item.sha256}":
            raise XmlCollectionRecipeError(f"inventory and similarity fingerprint disagree: {rel}")
        path = Path(item.path)
        # Explicitly reject symlinks and source substitution after inventory.
        if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
            raise XmlCollectionRecipeError(f"XML source is not a regular, non-symlink file: {rel}")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or resolved.relative_to(root).as_posix() != rel:
            raise XmlCollectionRecipeError(f"XML source path changed or escaped directory: {rel}")
        artifact = inspect_xml_structure(resolved, max_source_bytes=max_source_bytes)
        if artifact.source_sha256 != item.sha256:
            raise XmlCollectionRecipeError(f"XML source changed since inventory: {rel}")
        artifacts[rel] = artifact
    return review, artifacts


def inspect_xml_collection_recipe(
    inventory: XmlCollectionInventory, similarity: Mapping[str, Any],
    selected_paths: Iterable[str], *, acknowledge_variants: bool = False,
    record_root_canonical_path: str | None = None,
    max_source_bytes: int = 512 * 1024 ** 2,
) -> dict[str, Any]:
    """List record roots shared by reviewed sources and candidate fields at a chosen root.

    Caller must regenerate C1/C2 from current sources. No source values are extracted.
    """
    _, artifacts = _review_sources(inventory, similarity, selected_paths,
                                   acknowledge_variants, max_source_bytes=max_source_bytes)
    return _catalog_from_artifacts(inventory, artifacts, record_root_canonical_path,
                                   acknowledge_variants=acknowledge_variants)


def _catalog_from_artifacts(inventory: XmlCollectionInventory, artifacts: Mapping[str, Any],
                            record_root_canonical_path: str | None, *,
                            acknowledge_variants: bool) -> dict[str, Any]:
    all_profiles = [{p.canonical_path: p for p in art.element_profiles} for art in artifacts.values()]
    shared_roots = set.intersection(*(set(x) for x in all_profiles))
    options = []
    for path in shared_roots:
        profiles = [profiles[path] for profiles in all_profiles]
        options.append({
            "canonical_path": path,
            "display_path": profiles[0].path,
            "min_records": min(p.occurrence_count for p in profiles),
            "candidate_score": round(sum(p.record_candidate_score for p in profiles) / len(profiles), 4),
        })
    options.sort(key=lambda row: (-row["candidate_score"], -row["min_records"], row["canonical_path"]))
    chosen = (record_root_canonical_path or "").strip()
    if chosen and chosen not in shared_roots:
        raise XmlCollectionRecipeError("record root must exist in every selected XML source")

    selected = tuple(artifacts)
    catalog: dict[str, dict[str, Any]] = {}
    if chosen:
        for rel, artifact in artifacts.items():
            for field in build_xml_record_selection(artifact, chosen).fields:
                entry = catalog.setdefault(field.field_id, {
                    "field_id": field.field_id, "relative_path": field.relative_path,
                    "kind": field.kind, "canonical_path": field.canonical_path,
                    "present_in": [], "repeated_in": [], "optional_in": [],
                })
                entry["present_in"].append(rel)
                if field.repeated:
                    entry["repeated_in"].append(rel)
                if field.optional:
                    entry["optional_in"].append(rel)
    fields = []
    for field in sorted(catalog.values(), key=lambda row: row["field_id"]):
        field["missing_in"] = [rel for rel in selected if rel not in field["present_in"]]
        fields.append(field)
    return {
        "schema": "ml-lab.xml-collection-field-catalog@1",
        "root_path": inventory.root_path,
        "selected_sources": list(selected),
        "source_fingerprints": {rel: f"sha256:{art.source_sha256}" for rel, art in artifacts.items()},
        "acknowledge_variants": bool(acknowledge_variants),
        "record_root_options": options,
        "record_root_canonical_path": chosen or None,
        "fields": fields,
        "notes": ["Record root must be present in every source.",
                  "Fields absent from some sources require explicit missing-field approval.",
                  "Value-level extraction, preview, and CSV export belong to XML-C5."],
    }


def build_xml_collection_recipe(
    inventory: XmlCollectionInventory, similarity: Mapping[str, Any],
    selected_paths: Iterable[str], record_root_canonical_path: str,
    selected_field_ids: Iterable[str], *, acknowledge_variants: bool = False,
    allow_missing_fields: bool = False,
    rules: Iterable[Mapping[str, Any]] | None = None,
    max_source_bytes: int = 512 * 1024 ** 2,
) -> dict[str, Any]:
    """Validate one shared root/field/rule plan against every selected XML source.

    Returns a diagnostic record even when fields or rules fail validation; a bad
    source review or stale source raises instead. Output is never merge approval.
    """
    _, artifacts = _review_sources(inventory, similarity, selected_paths,
                                   acknowledge_variants, max_source_bytes=max_source_bytes)
    catalog = _catalog_from_artifacts(inventory, artifacts, record_root_canonical_path,
                                      acknowledge_variants=acknowledge_variants)
    root = catalog["record_root_canonical_path"]
    if not root:
        raise XmlCollectionRecipeError("select a shared record root")
    identifiers = [str(value) for value in selected_field_ids]
    if len(set(identifiers)) != len(identifiers):
        raise XmlCollectionRecipeError("duplicate field IDs in shared recipe")
    field_catalog = {row["field_id"]: row for row in catalog["fields"]}
    unknown = [field_id for field_id in identifiers if field_id not in field_catalog]
    if unknown:
        raise XmlCollectionRecipeError("unknown field IDs for shared record root: " + ", ".join(unknown[:5]))
    rule_map: dict[str, Mapping[str, Any]] = {}
    for rule in rules or ():
        if not isinstance(rule, Mapping):
            raise XmlCollectionRecipeError("repeated-branch rule must be an object")
        branch = str(rule.get("branch_canonical_path", ""))
        if not branch or branch in rule_map:
            raise XmlCollectionRecipeError("repeated-branch rule path must be unique and nonempty")
        rule_map[branch] = rule
    blocks: list[str] = []
    warnings: list[str] = []
    if not identifiers:
        blocks.append("Select at least one scalar field.")
    per_source: list[dict[str, Any]] = []
    required_branches: set[str] = set()
    all_sources = catalog["selected_sources"]
    # Re-evaluate each source's existing single-file plan, never inventing a
    # shared branch from path similarity alone.
    for rel in all_sources:
        artifact = artifacts[rel]
        source_fields = {field.field_id for field in build_xml_record_selection(artifact, root).fields}
        present = [field_id for field_id in identifiers if field_id in source_fields]
        missing = [field_id for field_id in identifiers if field_id not in source_fields]
        if missing and not allow_missing_fields:
            blocks.append(f"{rel}: {len(missing)} selected field(s) absent; explicitly allow missing fields.")
        if not present:
            blocks.append(f"{rel}: no selected fields present; cannot establish extraction.")
        # First discover branches without supplying rules, then apply the shared
        # rule set only to branches that exist in this source.
        preliminary = build_xml_collection_plan(artifact, root, present)
        active_branches = {branch.canonical_path for branch in preliminary.repeated_branches}
        required_branches.update(active_branches)
        specific_rules = [rule_map[branch] for branch in sorted(active_branches & rule_map.keys())]
        plan = build_xml_collection_plan(artifact, root, present, rules=specific_rules)
        issues = []
        for branch in plan.repeated_branches:
            if branch.rule is None or not branch.rule.valid:
                issues.append({"branch": branch.canonical_path,
                               "errors": list(branch.rule.errors) if branch.rule else ["Missing branch rule"]})
        if plan.unresolved_branch_count or plan.invalid_rule_count:
            blocks.append(f"{rel}: unresolved or invalid repeated-branch strategies.")
        for warning in plan.warnings:
            if warning not in warnings:
                warnings.append(warning)
        per_source.append({
            "relative_path": rel, "source_fingerprint": f"sha256:{artifact.source_sha256}",
            "record_root_occurrences": build_xml_record_selection(artifact, root).record_root_occurrence_count,
            "present_field_ids": present, "missing_field_ids": missing,
            "active_repeated_branches": sorted(active_branches), "rule_issues": issues,
            "ready_for_preview": plan.ready_for_preview and bool(present),
        })
    for branch in sorted(set(rule_map) - required_branches):
        blocks.append(f"Unused or unknown repeated-branch rule: {branch}")
    if any(field_catalog[fid]["missing_in"] for fid in identifiers):
        warnings.append("Optional/missing source fields require null alignment during future C5 extraction.")
    if any(field_catalog[fid]["repeated_in"] for fid in identifiers):
        warnings.append("Repeated-field value semantics must still be confirmed in XML-C5 previews.")
    ready = not blocks
    return {
        "schema": "ml-lab.xml-collection-recipe@1",
        "catalog_schema": catalog["schema"],
        "root_path": catalog["root_path"],
        "selected_sources": per_source,
        "record_root_canonical_path": root,
        "selected_field_ids": identifiers,
        "field_alignment": [field_catalog[fid] for fid in identifiers],
        "allow_missing_fields": bool(allow_missing_fields),
        "acknowledge_variants": bool(acknowledge_variants),
        "rules": [dict(rule_map[path]) for path in sorted(rule_map)],
        "required_repeated_branches": sorted(required_branches),
        "status": "ready_for_collection_preview" if ready else "blocked",
        "structural_recipe_validated": ready,
        "merge_authorized": False,
        "csv_generated": False,
        "blocking_reasons": blocks,
        "warnings": warnings,
        "notes": ["Only structural and repeated-branch configuration is validated.",
                  "XML-C5 must confirm actual value/type behavior and output schema before CSV export."],
    }


def save_xml_collection_recipe(recipe: Mapping[str, Any], destination: str | Path) -> Path:
    """Write a reviewed, structurally valid recipe JSON; never touch XML input."""
    if recipe.get("schema") != "ml-lab.xml-collection-recipe@1" or not recipe.get("structural_recipe_validated"):
        raise XmlCollectionRecipeError("only validated XML collection recipes may be exported")
    target = Path(destination).expanduser()
    if target.suffix.casefold() != ".json":
        target /= "xml_collection_recipe.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(recipe, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def main(argv: list[str] | None = None) -> int:
    """Run C4 collection recipe inspection from its standalone module entry point.

    This keeps CLI support without patching the central ml_lab.cli command tree.
    """
    import argparse

    from .xml_collection import inventory_xml_collection
    from .xml_similarity import compare_xml_collection

    parser = argparse.ArgumentParser(description="Inspect/validate shared XML recipe (no CSV export)")
    parser.add_argument("directory")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--source", action="append", required=True, help="Repeat XML source path relative to directory")
    parser.add_argument("--record-root", help="Canonical XML record root shared by selected sources")
    parser.add_argument("--field", action="append", default=[], help="Repeat canonical XML field ID")
    parser.add_argument("--rules-json", help="JSON list of repeated-branch rules")
    parser.add_argument("--acknowledge-variants", action="store_true")
    parser.add_argument("--allow-missing-fields", action="store_true")
    parser.add_argument("--min-shared-fraction", type=float, default=0.5)
    parser.add_argument("--max-files", type=int, default=80)
    parser.add_argument("--inspect", action="store_true", help="Display shared root/field choices as JSON")
    parser.add_argument("--output", default="analysis/data/xml_collection_recipe.json")
    args = parser.parse_args(argv)
    inventory = inventory_xml_collection(args.directory, recursive=args.recursive, max_files=args.max_files)
    report = compare_xml_collection(inventory, min_shared_fraction=args.min_shared_fraction)
    if args.inspect:
        catalog = inspect_xml_collection_recipe(
            inventory, report, args.source, acknowledge_variants=args.acknowledge_variants,
            record_root_canonical_path=args.record_root,
        )
        print(json.dumps(catalog, indent=2, sort_keys=True))
        return 0
    if not args.record_root or not args.field:
        parser.error("recipe validation requires --record-root and at least one --field; use --inspect to discover options")
    rules: list[dict[str, Any]] = []
    if args.rules_json:
        value = json.loads(Path(args.rules_json).read_text(encoding="utf-8"))
        if not isinstance(value, list):
            parser.error("--rules-json must contain a JSON list")
        rules = value
    recipe = build_xml_collection_recipe(
        inventory, report, args.source, args.record_root, args.field,
        acknowledge_variants=args.acknowledge_variants,
        allow_missing_fields=args.allow_missing_fields,
        rules=rules,
    )
    if recipe["status"] != "ready_for_collection_preview":
        print(json.dumps(recipe, indent=2, sort_keys=True))
        return 2
    destination = save_xml_collection_recipe(recipe, args.output)
    print(f"Validated XML collection recipe written to: {destination}")
    print("Structural recipe only; no rows extracted or CSV generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
