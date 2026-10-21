"""XML-C3: explicit, non-mutating collection selection and compatibility review.

The C2 classifications are evidence, not validated extraction schemas. A review
may be ready for recipe design without ever authorizing a multi-file export.
"""
from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


def _comparison_record(report: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Normalize the typed C2 artifact and older mapping snapshots."""
    if hasattr(report, "to_record"):
        report = report.to_record()
    if not isinstance(report, Mapping):
        raise ValueError("an XML-C2 structural similarity report is required")
    record = dict(report)
    if record.get("schema") != "ml-lab.xml-collection-similarity@1":
        raise ValueError("an XML-C2 structural similarity report is required")
    return record


def review_xml_collection(
    report: Mapping[str, Any] | Any,
    selected_paths: Iterable[str],
    *,
    acknowledge_variants: bool = False,
) -> dict[str, Any]:
    """Validate proposed collection membership against fresh C2 evidence.

    C2 is authoritative for structural evidence. This adapter accepts both the
    typed C2 artifact and serialized mapping snapshots, while preserving the
    stricter C3 rule that no similarity result itself authorizes a merge.
    """
    report = _comparison_record(report)
    selections = list(selected_paths)
    if len(selections) != len(set(selections)):
        raise ValueError("duplicate selected XML source paths")
    sources = report.get("sources", [])
    known = {source["relative_path"]: source for source in sources}
    unknown = [path for path in selections if path not in known]
    if unknown:
        raise ValueError("unknown XML collection source(s): " + ", ".join(unknown[:5]))

    selected = sorted(selections)
    selected_set = set(selected)
    omitted = sorted(set(known) - selected_set)
    valid_statuses = {"analyzed", "compared"}
    invalid = [path for path in selected if known[path].get("status") not in valid_statuses]
    pairs = report.get("pairwise")
    if pairs is None:
        pairs = report.get("pairs", [])
    selected_pairs = [pair for pair in pairs
                      if pair["left"] in selected_set and pair["right"] in selected_set]
    expected_pairs = len(selected) * (len(selected) - 1) // 2
    incompatible = [pair for pair in selected_pairs if pair.get("classification") == "incompatible"]
    exact_classes = {"exact", "exact_structure"}
    variant_classes = {"compatible_variant_candidate", "compatible_variant_review_required", "review_required"}
    variants = [pair for pair in selected_pairs if pair.get("classification") in variant_classes]
    missing_pairs = expected_pairs - len(selected_pairs)
    blocks: list[str] = []
    if len(selected) < 2:
        blocks.append("Select at least two XML sources to prepare a collection recipe.")
    if invalid:
        blocks.append("Unanalyzed or failed XML sources were selected: " + ", ".join(invalid[:5]))
    if missing_pairs:
        blocks.append(f"Missing comparison evidence for {missing_pairs} selected pair(s).")
    if incompatible:
        blocks.append(f"{len(incompatible)} incompatible source pair(s) are selected; revise membership.")
    if variants and not acknowledge_variants:
        blocks.append("Compatible-variant pairs require explicit acknowledgment before recipe design.")

    readiness = "blocked" if blocks else "ready_for_recipe_design"
    return {
        "schema": "ml-lab.xml-collection-review@1",
        "similarity_schema": report["schema"],
        "root_path": report["root_path"],
        "status": readiness,
        "merge_authorized": False,
        "extraction_recipe_validated": False,
        "acknowledge_variants": bool(acknowledge_variants),
        "selected": [{
            "relative_path": path,
            "source_fingerprint": known[path].get("source_fingerprint"),
            "status": known[path].get("status"),
            "root": known[path].get("root") or known[path].get("root_tag"),
            "record_candidates": known[path].get("record_candidates") or known[path].get("record_root_candidates", []),
        } for path in selected],
        "omitted": [{
            "relative_path": path,
            "status": known[path].get("status"),
            "reason": known[path].get("reason") or "not selected",
        } for path in omitted],
        "pairwise": selected_pairs,
        "summary": {
            "selected_count": len(selected),
            "omitted_count": len(omitted),
            "exact_pair_count": sum(p.get("classification") in exact_classes for p in selected_pairs),
            "variant_pair_count": len(variants),
            "incompatible_pair_count": len(incompatible),
            "missing_pair_count": max(0, missing_pairs),
        },
        "blocking_reasons": blocks,
        "notes": [
            "Review only: matching structure does not imply compatible extraction semantics.",
            "XML-C4 validates record roots and shared field mappings before extraction.",
            "No CSV consolidation, source alteration, or merge authorization occurs here.",
        ],
    }
