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
from typing import Any

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
    cardinality = {key for key in common if a.paths[key][1:3] != b.paths[key][1:3]}
    text_shape = {key for key in common if a.paths[key][3:] != b.paths[key][3:]}
    attributes_changed = {key for key in common if a.paths[key][0] != b.paths[key][0]}

    if a.root_tag != b.root_tag:
        status, reason = "incompatible", "different_expanded_root_qname"
    elif a.signature == b.signature:
        status, reason = "exact", "identical_structural_fingerprint"
    elif text_shape:
        status, reason = "review_required", "text_or_mixed_content_shape_differs"
    elif cardinality:
        status, reason = "review_required", "observed_cardinality_differs"
    elif similarity < threshold:
        status, reason = "incompatible", "insufficient_structure_overlap"
    elif not shared_candidates:
        status, reason = "review_required", "no_shared_record_root_evidence"
    else:
        status, reason = "compatible_variant_candidate", "high_overlap_shared_record_root"

    return {
        "left": a.relative_path,
        "right": b.relative_path,
        "classification": status,
        "reason": reason,
        "similarity_score": similarity,
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
            "text_shape_change_count": len(text_shape),
            "text_shape_changed_paths": _limited_paths(text_shape, max_diff_paths),
        },
        "merge_authorized": False,
    }


@dataclass(frozen=True)
class XmlCollectionComparison:
    root_path: str
    threshold: float
    sources: tuple[dict[str, Any], ...]
    pairs: tuple[dict[str, Any], ...]
    groups: tuple[dict[str, Any], ...]
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
            },
            "sources": list(self.sources),
            "pairs": list(self.pairs),
            "groups": list(self.groups),
            "notes": [
                "Similarity and observed cardinality are heuristics, not validated schemas.",
                "No implicit approval to merge files, infer source data types, or export a consolidated CSV.",
                "Groups are pairwise compatible candidates requiring subsequent user review.",
            ],
        }


def compare_xml_collection(
    inventory: XmlCollectionInventory,
    *,
    minimum_similarity: float = 0.80,
    max_sources: int = 100,
    max_unique_paths: int = 5000,
    max_file_bytes: int = 512 * 1024 ** 2,
    max_diff_paths: int = 20,
) -> XmlCollectionComparison:
    """Compare an XML-C1 inventory after rechecking each input fingerprint.

    Failed/stale files are reported, not treated as empty XML. A complete-link
    grouping policy prevents chains of individually similar but mutually
    incompatible files from entering one candidate group. No writes occur.
    """
    if not isinstance(inventory, XmlCollectionInventory):
        raise TypeError("expected an XML-C1 collection inventory")
    if not 0.0 < minimum_similarity <= 1.0:
        raise ValueError("minimum_similarity must be in (0, 1]")
    if any(value < 1 for value in (max_sources, max_unique_paths, max_file_bytes, max_diff_paths)):
        raise ValueError("comparison bounds must be positive")
    if max_sources > 200 or max_unique_paths > 100_000 or max_diff_paths > 100:
        raise ValueError("XML-C2 hard comparison safety limit exceeded")
    parsed_count = sum(item.status == "parsed" for item in inventory.files)
    if parsed_count > max_sources:
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
            comparisons_ok = all(
                pair_lookup[(existing.relative_path, shape.relative_path)]["classification"]
                in {"exact", "compatible_variant_candidate"} for existing in group
            )
            shared_roots = set(shape.record_roots)
            for existing in group:
                shared_roots.intersection_update(existing.record_roots)
            # Exact structure can be grouped without a repeated-record candidate;
            # variant groups must still have a shared candidate across *all* files.
            exact_group = all(existing.signature == shape.signature for existing in group)
            if comparisons_ok and (exact_group or shared_roots):
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
        groups.append({
            "group_id": f"group-{index:03d}",
            "classification": "singleton" if len(members) == 1 else ("exact" if exact else "compatible_variant_candidate"),
            "members": [member.relative_path for member in members],
            "shared_record_root_candidates": sorted(shared_roots)[:max_diff_paths],
            "merge_authorized": False,
        })
    return XmlCollectionComparison(inventory.root_path, minimum_similarity,
                                   tuple(sources), tuple(pairs), tuple(groups))


def save_xml_collection_comparison(comparison: XmlCollectionComparison, destination: str | Path) -> Path:
    """Persist derived comparison evidence only after explicit user request."""
    target = Path(destination).expanduser()
    if target.suffix.casefold() != ".json":
        target /= "xml_collection_similarity.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(comparison.to_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
