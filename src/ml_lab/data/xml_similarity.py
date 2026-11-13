"""XML-C2: bounded, read-only structural comparison of XML-C1 collection inventories.

Structural similarity is *evidence*, not permission to merge sources.  Prefixes are
ignored in favor of expanded QNames; values, occurrence counts, and inferred data
types are deliberately not used as authoritative schema constraints.
"""

from __future__ import annotations

import hashlib
import json
import stat
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator
from collections.abc import Mapping

from .xml_collection import XmlCollectionInventory
from .xml_ingest import XmlSafetyError, inspect_xml_structure
from .types import XmlStructureArtifact


@dataclass(frozen=True)
class _Shape:
    relative_path: str
    source_sha256: str
    root_tag: str
    signature: str
    paths: dict[str, tuple[tuple[str, ...], bool, bool, bool, bool]]
    record_roots: frozenset[str]
    occurrences: dict[str, int]
    text_occurrences: dict[str, int]
    attribute_occurrences: dict[str, dict[str, int]]
    # Per-path: sorted expanded attribute QNames, repeated, optional, has_text, mixed.

    def to_record(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "source_fingerprint": f"sha256:{self.source_sha256}",
            "root_tag": self.root_tag,
            "structure_fingerprint": self.signature,
            "path_count": len(self.paths),
            "attribute_count": sum(len(p[0]) for p in self.paths.values()),
            "record_root_candidates": sorted(self.record_roots),
            "type_evidence": "not_inferred",
        }


def _shape(relative: str, artifact: XmlStructureArtifact) -> _Shape:
    profiles = {
        item.canonical_path: (
            tuple(sorted(attr.name for attr in item.attributes)),
            item.repeated, item.optional,
            bool(item.text_occurrence_count), item.mixed_content,
        )
        for item in artifact.element_profiles
    }
    encoded = json.dumps(
        {"root": artifact.root_tag, "profiles": sorted(profiles.items())},
        separators=(",", ":"), ensure_ascii=False,
    ).encode("utf-8")
    return _Shape(
        relative_path=relative,
        source_sha256=artifact.source_sha256,
        root_tag=artifact.root_tag,
        signature="sha256:" + hashlib.sha256(encoded).hexdigest(),
        paths=profiles,
        record_roots=frozenset(item.canonical_path for item in artifact.record_candidates),
        occurrences={p.canonical_path: p.occurrence_count for p in artifact.element_profiles},
        text_occurrences={p.canonical_path: p.text_occurrence_count for p in artifact.element_profiles},
        attribute_occurrences={p.canonical_path: {a.name: a.occurrence_count for a in p.attributes}
                               for p in artifact.element_profiles},
    )


def _limited_paths(paths: set[str], limit: int) -> list[str]:
    return sorted(paths)[:limit]


def _pair(a: _Shape, b: _Shape, threshold: float, max_diff_paths: int) -> dict[str, Any]:
    a_paths, b_paths = set(a.paths), set(b.paths)
    common = a_paths & b_paths
    union = a_paths | b_paths
    paths_score = len(common) / len(union) if union else 1.0
    a_attributes = {(key, name) for key, features in a.paths.items() for name in features[0]}
    b_attributes = {(key, name) for key, features in b.paths.items() for name in features[0]}
    attributes_union = a_attributes | b_attributes
    attributes_score = len(a_attributes & b_attributes) / len(attributes_union) if attributes_union else 1.0
    similarity = round(0.8 * paths_score + 0.2 * attributes_score, 6)
    shared_candidates = sorted(a.record_roots & b.record_roots)
    # Optionality can differ solely because values/paths are absent in some
    # records. That is nullable evidence, not an automatic grouping conflict.
    # Repetition changes affect scalar/child-table semantics and need review.
    cardinality = {key for key in common if a.paths[key][1] != b.paths[key][1]}
    optionality = {key for key in common if a.paths[key][2] != b.paths[key][2]}
    # Sparse/empty/nil text is a per-record observation, not a structural
    # conflict. Mixed content, on the other hand, changes extraction semantics.
    mixed_content = {key for key in common if a.paths[key][4] != b.paths[key][4]}
    text_presence = {key for key in common if a.paths[key][3] != b.paths[key][3]}
    attributes_changed = {key for key in common if a.paths[key][0] != b.paths[key][0]}

    if a.root_tag != b.root_tag:
        status, reason = "incompatible", "different_expanded_root_qname"
    elif a.signature == b.signature:
        status, reason = "exact", "identical_structural_fingerprint"
    elif mixed_content:
        status, reason = "review_required", "mixed_content_shape_differs"
    elif cardinality:
        status, reason = "review_required", "observed_cardinality_differs"
    elif not shared_candidates:
        status, reason = "review_required", "no_shared_record_root_evidence"
    else:
        # Path and attribute absence (and differing text presence) is evidence
        # for optional/null fields. Similarity is diagnostic, not a merge gate.
        status, reason = "compatible_variant_candidate", "null_tolerant_shared_record_root"

    return {
        "left": a.relative_path,
        "right": b.relative_path,
        "classification": status,
        "reason": reason,
        "similarity_score": similarity,
        "below_similarity_threshold": similarity < threshold,
        "missing_fields_null_tolerated": status == "compatible_variant_candidate",
        "path_jaccard": round(paths_score, 6),
        "attribute_jaccard": round(attributes_score, 6),
        "shared_record_root_candidates": shared_candidates[:max_diff_paths],
        "differences": {
            "left_only_path_count": len(a_paths - b_paths),
            "left_only_paths": _limited_paths(a_paths - b_paths, max_diff_paths),
            "right_only_path_count": len(b_paths - a_paths),
            "right_only_paths": _limited_paths(b_paths - a_paths, max_diff_paths),
            "attribute_change_count": len(attributes_changed),
            "attribute_changed_paths": _limited_paths(attributes_changed, max_diff_paths),
            "cardinality_change_count": len(cardinality),
            "cardinality_changed_paths": _limited_paths(cardinality, max_diff_paths),
            "optional_presence_change_count": len(optionality),
            "optional_presence_changed_paths": _limited_paths(optionality, max_diff_paths),
            "text_shape_change_count": len(mixed_content),
            "text_shape_changed_paths": _limited_paths(mixed_content, max_diff_paths),
            "text_presence_change_count": len(text_presence),
            "text_presence_changed_paths": _limited_paths(text_presence, max_diff_paths),
        },
        "merge_authorized": False,
    }


@dataclass(frozen=True)
class XmlCollectionComparison(Mapping[str, Any]):
    root_path: str
    threshold: float
    sources: tuple[dict[str, Any], ...]
    pairs: tuple[dict[str, Any], ...]
    groups: tuple[dict[str, Any], ...]
    # Additive union metadata; legacy positional constructor/API remains valid.
    union_structure: dict[str, Any] | None = None
    schema: str = "ml-lab.xml-collection-similarity@1"

    def to_record(self) -> dict[str, Any]:
        counts = {name: sum(item["classification"] == name for item in self.pairs)
                  for name in ("exact", "compatible_variant_candidate", "review_required", "incompatible")}
        return {
            "schema": self.schema,
            "root_path": self.root_path,
            "settings": {"minimum_similarity": self.threshold},
            "summary": {
                "source_count": len(self.sources),
                "compared_source_count": sum(item["status"] == "compared" for item in self.sources),
                "excluded_source_count": sum(item["status"] != "compared" for item in self.sources),
                "pair_count": len(self.pairs),
                "pair_classifications": counts,
                "group_count": len(self.groups),
                "merge_authorized": False,
                "consolidation_status": "not_available",
                "union_structure_available": self.union_structure is not None,
            },
            "sources": list(self.sources),
            "pairs": list(self.pairs),
            "groups": list(self.groups),
            "union_structure": self.union_structure,
            "notes": [
                "Similarity and observed cardinality are heuristics, not validated schemas.",
                "No implicit approval to merge files, infer source data types, or export a consolidated CSV.",
                "Groups are pairwise compatible candidates requiring subsequent user review.",
                "Missing element/attribute paths and empty text are null-tolerated; minimum similarity is advisory.",
                "The union is an observed structure, not a validated extraction recipe or schema.",
            ],
        }

    # C3-C7 historically consumed the comparison as a mapping while C2 now
    # exposes a typed artifact. Preserve both interfaces at the boundary so
    # collection review/recipe/export code can consume one authoritative C2
    # representation without duplicating conversion logic.
    def __getitem__(self, key: str) -> Any:
        return self.to_record()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.to_record())

    def __len__(self) -> int:
        return len(self.to_record())



UNION_SCHEMA = "ml-lab.xml-collection-union-structure@1"


def _observed_union(shapes: list[_Shape], groups: list[dict[str, Any]]) -> dict[str, Any]:
    """Report one canonical source-linked union, with non-copying group views.

    Namespaces are represented by the same escaped expanded-QName paths as the
    ML-UW-18 null field catalog. A path absent in a source is *optional* in this
    observed union; we do not insert elements into the XML or infer record values.
    Count maps retain exact numbers reported by the inspected source profiles.
    """
    by_path: dict[str, dict[str, Any]] = {}
    for shape in shapes:
        for path, (attribute_names, repeated, optional, has_text, mixed) in shape.paths.items():
            entry = by_path.setdefault(path, {
                "canonical_path": path,
                "root_tag": shape.root_tag,
                "parent_canonical_path": path.rpartition("/")[0] or None,
                "present_in": [],
                "occurrences_by_source": {},
                "text_occurrences_by_source": {},
                "repeated_in": [],
                "optional_in": [],
                "mixed_content_in": [],
                "attributes": {},
            })
            entry["present_in"].append(shape.relative_path)
            entry["occurrences_by_source"][shape.relative_path] = shape.occurrences[path]
            entry["text_occurrences_by_source"][shape.relative_path] = shape.text_occurrences[path]
            if repeated:
                entry["repeated_in"].append(shape.relative_path)
            if optional:
                entry["optional_in"].append(shape.relative_path)
            if mixed:
                entry["mixed_content_in"].append(shape.relative_path)
            for name in attribute_names:
                field = entry["attributes"].setdefault(name, {
                    "field_id": f"attribute:{path}@{name}",
                    "attribute_qname": name,
                    "present_in": [],
                    "occurrences_by_source": {},
                })
                field["present_in"].append(shape.relative_path)
                field["occurrences_by_source"][shape.relative_path] = shape.attribute_occurrences[path][name]
    all_sources = sorted(shape.relative_path for shape in shapes)
    paths = []
    for path, entry in sorted(by_path.items()):
        entry["present_in"].sort()
        entry["source_count"] = len(entry["present_in"])
        entry["missing_source_count"] = len(all_sources) - entry["source_count"]
        entry["total_element_occurrences"] = sum(entry["occurrences_by_source"].values())
        entry["total_text_occurrences"] = sum(entry["text_occurrences_by_source"].values())
        entry["attributes"] = [
            {**item, "present_in": sorted(item["present_in"]),
             "source_count": len(item["present_in"]),
             "missing_source_count": len(all_sources) - len(item["present_in"]),
             "total_occurrences": sum(item["occurrences_by_source"].values())}
            for _, item in sorted(entry["attributes"].items())
        ]
        paths.append(entry)
    source_roots: dict[str, list[str]] = {}
    for shape in shapes:
        source_roots.setdefault(shape.root_tag, []).append(shape.relative_path)
    # Build views in one pass rather than scanning the entire union for every
    # candidate group (which would scale poorly with many singleton groups).
    group_by_member = {member: group["group_id"] for group in groups
                       for member in group["members"]}
    path_ids_by_group: dict[str, list[str]] = {group["group_id"]: [] for group in groups}
    optional_ids_by_group: dict[str, list[str]] = {group["group_id"]: [] for group in groups}
    group_sizes = {group["group_id"]: len(group["members"]) for group in groups}
    for entry in paths:
        present_counts: dict[str, int] = {}
        for source_name in entry["present_in"]:
            group_id = group_by_member[source_name]
            present_counts[group_id] = present_counts.get(group_id, 0) + 1
        for group_id in sorted(present_counts):
            path_ids_by_group[group_id].append(entry["canonical_path"])
            if present_counts[group_id] < group_sizes[group_id]:
                optional_ids_by_group[group_id].append(entry["canonical_path"])
    root_by_source = {shape.relative_path: shape.root_tag for shape in shapes}
    return {
        "schema": UNION_SCHEMA,
        "source_paths": all_sources,
        "source_count": len(all_sources),
        "root_partitions": [{"root_tag": root, "members": sorted(members)}
                            for root, members in sorted(source_roots.items())],
        "path_count": len(paths),
        "paths": paths,
        # Group views index the canonical path inventory without duplicating it.
        "groups": [
            {"group_id": group["group_id"], "members": list(group["members"]),
             "root_tag": root_by_source[group["members"][0]],
             "path_ids": path_ids_by_group[group["group_id"]],
             "optional_path_ids": optional_ids_by_group[group["group_id"]],
             "shared_record_root_candidates": list(group["shared_record_root_candidates"]),
             "merge_authorized": False}
            for group in groups
        ],
        "merge_authorized": False,
        "null_policy": "missing_paths_are_optional; missing_values_are_not_schema_conflicts",
        "notes": ["Occurrence counts are observed, not inferred or imputed.",
                  "Absent path values and Comments provenance are materialized in ML-UW-20/21.",
                  "Root QName differences remain partitioned; mixed content/cardinality require review."],
    }



def compare_xml_collection(
    inventory: XmlCollectionInventory,
    *,
    minimum_similarity: float = 0.80,
    min_shared_fraction: float | None = None,
    max_sources: int | None = None,
    max_unique_paths: int = 5000,
    max_file_bytes: int = 512 * 1024 ** 2,
    max_diff_paths: int = 20,
) -> XmlCollectionComparison:
    """Compare an XML-C1 inventory after rechecking each input fingerprint.

    Failed/stale files are reported, not treated as empty XML. A complete-link
    grouping policy prevents chains of individually similar but mutually
    incompatible files from entering one candidate group. No writes occur.
    All parsed sources are compared unless a caller explicitly supplies
    ``max_sources``; time and report size grow quadratically with source count.
    """
    if not isinstance(inventory, XmlCollectionInventory):
        raise TypeError("expected an XML-C1 collection inventory")
    if min_shared_fraction is not None:
        if minimum_similarity != 0.80 and minimum_similarity != min_shared_fraction:
            raise ValueError("minimum_similarity and min_shared_fraction disagree")
        minimum_similarity = min_shared_fraction
    if not 0.0 < minimum_similarity <= 1.0:
        raise ValueError("minimum_similarity must be in (0, 1]")
    if any(value < 1 for value in (max_unique_paths, max_file_bytes, max_diff_paths)) or (max_sources is not None and max_sources < 1):
        raise ValueError("comparison bounds must be positive")
    if max_unique_paths > 100_000 or max_diff_paths > 100:
        raise ValueError("XML-C2 hard comparison safety limit exceeded")
    parsed_count = sum(item.status == "parsed" for item in inventory.files)
    if max_sources is not None and parsed_count > max_sources:
        raise ValueError(f"XML comparison source limit exceeded ({max_sources}); reduce selection")
    root = Path(inventory.root_path).resolve()
    shapes: list[_Shape] = []
    sources: list[dict[str, Any]] = []
    for item in inventory.files:
        if item.status != "parsed":
            sources.append({"relative_path": item.relative_path, "status": "excluded",
                            "reason": item.error or "inventory_parse_failed"})
            continue
        try:
            path = Path(item.path)
            if path.is_symlink():
                raise XmlSafetyError("source is a symlink")
            resolved = path.resolve(strict=True)
            if not resolved.is_relative_to(root):
                raise XmlSafetyError("source escapes inventory directory")
            before = resolved.stat()
            if not stat.S_ISREG(before.st_mode) or before.st_size != item.size_bytes:
                raise XmlSafetyError("source metadata differs from inventory")
            if before.st_size > max_file_bytes:
                raise XmlSafetyError("XML comparison size limit exceeded")
            artifact = inspect_xml_structure(resolved, max_source_bytes=max_file_bytes,
                                             max_unique_paths=max_unique_paths)
            after = resolved.stat()
            if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                raise XmlSafetyError("source changed during comparison")
            if artifact.source_sha256 != item.sha256:
                raise XmlSafetyError("source fingerprint differs from XML-C1 inventory")
            shape = _shape(item.relative_path, artifact)
            shapes.append(shape)
            sources.append({"status": "compared", **shape.to_record()})
        except (OSError, XmlSafetyError, ET.ParseError, LookupError, UnicodeError, ValueError) as exc:
            sources.append({"relative_path": item.relative_path, "status": "excluded",
                            "reason": f"{type(exc).__name__}: {exc}"})
    shapes.sort(key=lambda item: item.relative_path)
    pairs: list[dict[str, Any]] = []
    pair_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for i, left in enumerate(shapes):
        for right in shapes[i + 1:]:
            result = _pair(left, right, minimum_similarity, max_diff_paths)
            pairs.append(result)
            pair_lookup[(left.relative_path, right.relative_path)] = result

    candidate_groups: list[list[_Shape]] = []
    for shape in shapes:
        for group in candidate_groups:
            comparisons = [pair_lookup[(existing.relative_path, shape.relative_path)]
                           for existing in group]
            # An absent auto-detected record candidate is inconclusive for
            # single-record XML files. Retain a *review-required* group rather
            # than fragmenting it only because optional paths differ. Never
            # group known repetition/mixed-content/root conflicts.
            comparisons_ok = all(
                pair["classification"] in {"exact", "compatible_variant_candidate"}
                or (pair["classification"] == "review_required" and
                    pair["reason"] == "no_shared_record_root_evidence")
                for pair in comparisons
            )
            shared_roots = set(shape.record_roots)
            for existing in group:
                shared_roots.intersection_update(existing.record_roots)
            if comparisons_ok:
                group.append(shape)
                break
        else:
            candidate_groups.append([shape])
    groups: list[dict[str, Any]] = []
    for index, members in enumerate(candidate_groups, start=1):
        exact = len({member.signature for member in members}) == 1
        shared_roots = set(members[0].record_roots)
        for member in members[1:]:
            shared_roots.intersection_update(member.record_roots)
        needs_root_review = any(
            pair_lookup[(left.relative_path, right.relative_path)]["reason"] == "no_shared_record_root_evidence"
            for i, left in enumerate(members) for right in members[i + 1:]
        )
        groups.append({
            "group_id": f"group-{index:03d}",
            "classification": ("singleton" if len(members) == 1 else
                               "exact" if exact else
                               "review_required" if needs_root_review else
                               "compatible_variant_candidate"),
            "members": [member.relative_path for member in members],
            "shared_record_root_candidates": sorted(shared_roots)[:max_diff_paths],
            "merge_authorized": False,
        })
    union = _observed_union(shapes, groups)
    return XmlCollectionComparison(inventory.root_path, minimum_similarity,
                                   tuple(sources), tuple(pairs), tuple(groups), union)


def save_xml_collection_comparison(comparison: XmlCollectionComparison, destination: str | Path) -> Path:
    """Persist derived comparison evidence only after explicit user request."""
    target = Path(destination).expanduser()
    if target.suffix.casefold() != ".json":
        target /= "xml_collection_similarity.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(comparison.to_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
