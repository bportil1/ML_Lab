from __future__ import annotations

import codecs
import hashlib
import re
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from .types import (
    XmlAttributeProfile,
    XmlElementProfile,
    XmlNamespace,
    XmlRecordCandidate,
    XmlStructureArtifact,
)


_MAX_XML_ELEMENTS = 2_000_000
_MAX_XML_DEPTH = 512
_MAX_XML_SOURCE_BYTES = 512 * 1024 * 1024
_MAX_XML_UNIQUE_PATHS = 100_000
_MAX_XML_NAMESPACES = 10_000
_MAX_ATTRIBUTE_VALUE_SAMPLES = 10_000
_MAX_TEXT_VALUE_SAMPLES = 10_000
_SCAN_CHUNK_BYTES = 64 * 1024
_XML_DECL_RE = re.compile(br"<\?xml\s+[^>]*encoding\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)
_IDENTIFIER_NAMES = {"id", "identifier", "uuid", "guid", "key", "uid"}
_RECORD_CANDIDATE_THRESHOLD = 0.40


class XmlSafetyError(ValueError):
    """Raised when XML input violates ML Lab's local safe-ingestion policy."""


@dataclass
class _AttributeStats:
    occurrence_count: int = 0
    sampled_value_count: int = 0
    distinct_values: set[str] = field(default_factory=set)


@dataclass
class _ElementStats:
    tag: str
    path_key: tuple[str, ...]
    occurrence_count: int = 0
    parents_with_element: int = 0
    positive_min_per_parent: int | None = None
    max_per_parent: int = 0
    text_occurrence_count: int = 0
    text_sampled_value_count: int = 0
    distinct_text_values: set[str] = field(default_factory=set)
    mixed_content_occurrence_count: int = 0
    child_keys: list[tuple[str, ...]] = field(default_factory=list)
    child_key_set: set[tuple[str, ...]] = field(default_factory=set)
    attributes: dict[str, _AttributeStats] = field(default_factory=dict)


@dataclass
class _Frame:
    path_key: tuple[str, ...]
    saw_child_tail_text: bool = False
    child_counts: Counter[tuple[str, ...]] = field(default_factory=Counter)


def _detect_xml_encoding(path: Path) -> tuple[str, bool]:
    with path.open("rb") as handle:
        prefix = handle.read(1024)
    if prefix.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig", prefix[3:].lstrip().startswith(b"<?xml")
    if prefix.startswith((b"\xff\xfe\x00\x00", b"\x00\x00\xfe\xff")):
        return "utf-32", True
    if prefix.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16", True
    match = _XML_DECL_RE.search(prefix)
    if match:
        try:
            return match.group(1).decode("ascii").strip(), True
        except UnicodeDecodeError:
            return "utf-8", True
    return "utf-8", prefix.lstrip().startswith(b"<?xml")


def _reject_unsafe_declarations(path: Path, encoding: str) -> None:
    """Reject DTD/entity declarations before handing bytes to ElementTree."""

    try:
        decoder = codecs.getincrementaldecoder(encoding)(errors="ignore")
    except LookupError as exc:
        raise XmlSafetyError(f"unsupported XML encoding declaration: {encoding}") from exc

    carry = ""
    raw_carry = b""
    with path.open("rb") as handle:
        while True:
            raw = handle.read(_SCAN_CHUNK_BYTES)
            if not raw:
                text = decoder.decode(b"", final=True)
                final = True
            else:
                raw_searchable = (raw_carry + raw).replace(b"\x00", b"").upper()
                if b"<!DOCTYPE" in raw_searchable or b"<!ENTITY" in raw_searchable:
                    raise XmlSafetyError("DTD/entity declarations are not allowed in XML Data Lab ingestion")
                raw_carry = raw_searchable[-64:]
                text = decoder.decode(raw, final=False)
                final = False
            searchable = (carry + text).upper()
            if "<!DOCTYPE" in searchable or "<!ENTITY" in searchable:
                raise XmlSafetyError("DTD/entity declarations are not allowed in XML Data Lab ingestion")
            carry = searchable[-64:]
            if final:
                break


def _split_qname(tag: str) -> tuple[str | None, str]:
    if tag.startswith("{") and "}" in tag:
        uri, local = tag[1:].split("}", 1)
        return uri, local
    return None, tag


def _preferred_prefix(namespaces: list[XmlNamespace], uri: str | None) -> str | None:
    if uri is None:
        return None
    for item in namespaces:
        if item.uri == uri:
            return item.prefix
    return None


def _display_qname(tag: str, namespaces: list[XmlNamespace]) -> str:
    uri, local = _split_qname(tag)
    prefix = _preferred_prefix(namespaces, uri)
    if prefix:
        return f"{prefix}:{local}"
    return local


def _display_path(path_key: tuple[str, ...], namespaces: list[XmlNamespace]) -> str:
    return "/" + "/".join(_display_qname(tag, namespaces) for tag in path_key)


def _canonical_path(path_key: tuple[str, ...]) -> str:
    # JSON-Pointer style escaping keeps expanded QNames round-trippable even when
    # namespace URIs contain '/' or '~'. Display paths remain human-readable.
    return "/" + "/".join(tag.replace("~", "~0").replace("/", "~1") for tag in path_key)


def _looks_like_identifier_name(local_name: str) -> bool:
    normalized = local_name.strip().lower().replace("-", "_")
    return normalized in _IDENTIFIER_NAMES or normalized.endswith("_id")


def _attribute_profiles(
    stats: _ElementStats,
    namespaces: list[XmlNamespace],
) -> tuple[XmlAttributeProfile, ...]:
    profiles: list[XmlAttributeProfile] = []
    for name, attr in stats.attributes.items():
        uri, local = _split_qname(name)
        sample_count = attr.sampled_value_count
        distinct_count = len(attr.distinct_values)
        presence_rate = attr.occurrence_count / stats.occurrence_count if stats.occurrence_count else 0.0
        distinct_rate = distinct_count / sample_count if sample_count else 0.0
        likely_identifier = (
            _looks_like_identifier_name(local)
            and presence_rate >= 0.80
            and sample_count > 0
            and distinct_rate >= 0.95
        )
        profiles.append(
            XmlAttributeProfile(
                name=name,
                local_name=local,
                namespace_uri=uri,
                prefix=_preferred_prefix(namespaces, uri),
                occurrence_count=attr.occurrence_count,
                element_occurrence_count=stats.occurrence_count,
                presence_rate=round(presence_rate, 6),
                sampled_value_count=sample_count,
                distinct_sample_count=distinct_count,
                distinct_sample_rate=round(distinct_rate, 6),
                likely_identifier=likely_identifier,
            )
        )
    return tuple(profiles)




def _text_identifier_evidence(stats: _ElementStats) -> tuple[bool, float]:
    sample_count = stats.text_sampled_value_count
    distinct_count = len(stats.distinct_text_values)
    distinct_rate = distinct_count / sample_count if sample_count else 0.0
    presence_rate = stats.text_occurrence_count / stats.occurrence_count if stats.occurrence_count else 0.0
    local = _split_qname(stats.tag)[1]
    likely = (
        _looks_like_identifier_name(local)
        and presence_rate >= 0.80
        and sample_count > 0
        and distinct_rate >= 0.95
    )
    return likely, distinct_rate


def _candidate_score(
    *,
    depth: int,
    occurrence_count: int,
    repeated: bool,
    has_children: bool,
    has_likely_identifier: bool,
) -> tuple[float, tuple[str, ...]]:
    if depth <= 1 or occurrence_count < 2:
        return 0.0, ()

    score = 0.0
    reasons: list[str] = []
    score += 0.25
    reasons.append("occurs_multiple_times")
    if repeated:
        score += 0.35
        reasons.append("repeats_within_parent")
    if has_children:
        score += 0.15
        reasons.append("has_child_structure")
    if has_likely_identifier:
        score += 0.20
        reasons.append("has_likely_identifier")
    if depth > 1:
        score += 0.05
        reasons.append("non_root_structure")
    return round(min(score, 1.0), 3), tuple(reasons)


def inspect_xml_structure(
    path: str | Path,
    *,
    source_sha256: str | None = None,
    max_elements: int = _MAX_XML_ELEMENTS,
    max_depth: int = _MAX_XML_DEPTH,
    max_source_bytes: int = _MAX_XML_SOURCE_BYTES,
    max_unique_paths: int = _MAX_XML_UNIQUE_PATHS,
    max_namespaces: int = _MAX_XML_NAMESPACES,
    max_attribute_value_samples: int = _MAX_ATTRIBUTE_VALUE_SAMPLES,
    max_text_value_samples: int = _MAX_TEXT_VALUE_SAMPLES,
) -> XmlStructureArtifact:
    """Safely parse XML and produce a bounded structural-analysis artifact.

    Sprint 2 extends the ingestion artifact with path-level hierarchy, attribute,
    cardinality, optional/repeated branch, likely-ID, and record-candidate analysis.
    Parsing remains streaming and source files remain immutable.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"not a file: {source}")
    if max_elements < 1:
        raise ValueError("max_elements must be at least 1")
    if max_depth < 1:
        raise ValueError("max_depth must be at least 1")
    if max_source_bytes < 1:
        raise ValueError("max_source_bytes must be at least 1")
    if max_unique_paths < 1:
        raise ValueError("max_unique_paths must be at least 1")
    if max_namespaces < 1:
        raise ValueError("max_namespaces must be at least 1")
    source_size = source.stat().st_size
    if source_size > max_source_bytes:
        raise XmlSafetyError(f"XML source size limit exceeded ({max_source_bytes:,} bytes)")
    if max_attribute_value_samples < 1:
        raise ValueError("max_attribute_value_samples must be at least 1")
    if max_text_value_samples < 1:
        raise ValueError("max_text_value_samples must be at least 1")

    if source_sha256 is None:
        digest = hashlib.sha256()
        with source.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        source_sha256 = digest.hexdigest()

    encoding, has_declaration = _detect_xml_encoding(source)
    _reject_unsafe_declarations(source, encoding)

    namespaces: list[XmlNamespace] = []
    namespace_pairs: set[tuple[str, str]] = set()
    root_tag: str | None = None
    root_attributes: tuple[str, ...] = ()
    top_level_tags: list[str] = []
    top_level_seen: set[str] = set()
    top_level_child_count = 0
    depth = 0
    observed_max_depth = 0
    element_count = 0
    frames: list[_Frame] = []
    stats_by_key: dict[tuple[str, ...], _ElementStats] = {}
    path_order: list[tuple[str, ...]] = []

    with source.open("rb") as handle:
        parser = ET.iterparse(handle, events=("start", "end", "start-ns"))
        for event, value in parser:
            if event == "start-ns":
                prefix, uri = value
                pair = (prefix or "", uri)
                if pair not in namespace_pairs:
                    if len(namespace_pairs) >= max_namespaces:
                        raise XmlSafetyError(f"XML namespace limit exceeded ({max_namespaces:,})")
                    namespace_pairs.add(pair)
                    namespaces.append(XmlNamespace(prefix=pair[0], uri=pair[1]))
                continue

            element = value
            if event == "start":
                depth += 1
                element_count += 1
                observed_max_depth = max(observed_max_depth, depth)
                if element_count > max_elements:
                    raise XmlSafetyError(f"XML element limit exceeded ({max_elements:,})")
                if depth > max_depth:
                    raise XmlSafetyError(f"XML nesting depth limit exceeded ({max_depth})")

                tag = str(element.tag)
                path_key = frames[-1].path_key + (tag,) if frames else (tag,)

                stats = stats_by_key.get(path_key)
                if stats is None:
                    if len(stats_by_key) >= max_unique_paths:
                        raise XmlSafetyError(f"XML unique element path limit exceeded ({max_unique_paths:,})")
                    stats = _ElementStats(tag=tag, path_key=path_key)
                    stats_by_key[path_key] = stats
                    path_order.append(path_key)
                stats.occurrence_count += 1

                if frames:
                    parent = frames[-1]
                    parent.child_counts[path_key] += 1
                    parent_stats = stats_by_key[parent.path_key]
                    if path_key not in parent_stats.child_key_set:
                        parent_stats.child_key_set.add(path_key)
                        parent_stats.child_keys.append(path_key)

                for attr_name, attr_value in element.attrib.items():
                    attr_key = str(attr_name)
                    attr_stats = stats.attributes.setdefault(attr_key, _AttributeStats())
                    attr_stats.occurrence_count += 1
                    if attr_stats.sampled_value_count < max_attribute_value_samples:
                        attr_stats.sampled_value_count += 1
                        attr_stats.distinct_values.add(str(attr_value))

                if depth == 1:
                    root_tag = tag
                    root_attributes = tuple(str(name) for name in element.attrib)
                elif depth == 2:
                    top_level_child_count += 1
                    if tag not in top_level_seen:
                        top_level_seen.add(tag)
                        top_level_tags.append(tag)

                frames.append(_Frame(path_key=path_key))
                continue

            frame = frames.pop()
            stats = stats_by_key[frame.path_key]
            leading_text = bool(element.text is not None and element.text.strip())
            if leading_text and frame.child_counts or frame.saw_child_tail_text:
                stats.mixed_content_occurrence_count += 1
            if element.text is not None and element.text.strip():
                stats.text_occurrence_count += 1
                if stats.text_sampled_value_count < max_text_value_samples:
                    text_value = element.text.strip()
                    stats.text_sampled_value_count += 1
                    stats.distinct_text_values.add(text_value)
            if frames and element.tail is not None and element.tail.strip():
                frames[-1].saw_child_tail_text = True
            for child_key, count in frame.child_counts.items():
                child_stats = stats_by_key[child_key]
                child_stats.parents_with_element += 1
                if child_stats.positive_min_per_parent is None:
                    child_stats.positive_min_per_parent = count
                else:
                    child_stats.positive_min_per_parent = min(child_stats.positive_min_per_parent, count)
                child_stats.max_per_parent = max(child_stats.max_per_parent, count)
            depth -= 1
            element.clear()

    if root_tag is None:
        raise ET.ParseError("XML document has no root element")

    profiles: list[XmlElementProfile] = []
    for path_key in path_order:
        stats = stats_by_key[path_key]
        parent_key = path_key[:-1] if len(path_key) > 1 else None
        parent_occurrences = stats_by_key[parent_key].occurrence_count if parent_key else None
        attributes = _attribute_profiles(stats, namespaces)
        likely_identifiers = tuple(item.name for item in attributes if item.likely_identifier)
        likely_identifier_text, text_distinct_rate = _text_identifier_evidence(stats)
        child_has_likely_identifier = any(
            _text_identifier_evidence(stats_by_key[child_key])[0]
            for child_key in stats.child_keys
        )
        if parent_occurrences is None:
            parents_with_element = None
            min_per_parent = None
            max_per_parent = None
            mean_per_parent = None
            repeated = False
            optional = False
            parent_path = None
        else:
            parents_with_element = stats.parents_with_element
            min_per_parent = (
                stats.positive_min_per_parent
                if stats.parents_with_element == parent_occurrences
                else 0
            )
            max_per_parent = stats.max_per_parent
            mean_per_parent = stats.occurrence_count / parent_occurrences if parent_occurrences else 0.0
            repeated = max_per_parent > 1
            optional = stats.parents_with_element < parent_occurrences
            parent_path = _display_path(parent_key, namespaces)

        score, reasons = _candidate_score(
            depth=len(path_key),
            occurrence_count=stats.occurrence_count,
            repeated=repeated,
            has_children=bool(stats.child_keys),
            has_likely_identifier=bool(likely_identifiers) or child_has_likely_identifier,
        )
        profile = XmlElementProfile(
            path=_display_path(path_key, namespaces),
            canonical_path=_canonical_path(path_key),
            tag=stats.tag,
            local_name=_split_qname(stats.tag)[1],
            namespace_uri=_split_qname(stats.tag)[0],
            prefix=_preferred_prefix(namespaces, _split_qname(stats.tag)[0]),
            depth=len(path_key),
            occurrence_count=stats.occurrence_count,
            parent_path=parent_path,
            parent_occurrence_count=parent_occurrences,
            parents_with_element=parents_with_element,
            min_per_parent=min_per_parent,
            max_per_parent=max_per_parent,
            mean_per_parent=round(mean_per_parent, 6) if mean_per_parent is not None else None,
            repeated=repeated,
            optional=optional,
            text_occurrence_count=stats.text_occurrence_count,
            text_presence_rate=round(stats.text_occurrence_count / stats.occurrence_count, 6)
            if stats.occurrence_count
            else 0.0,
            text_sampled_value_count=stats.text_sampled_value_count,
            text_distinct_sample_count=len(stats.distinct_text_values),
            text_distinct_sample_rate=round(text_distinct_rate, 6),
            likely_identifier_text=likely_identifier_text,
            mixed_content_occurrence_count=stats.mixed_content_occurrence_count,
            mixed_content=stats.mixed_content_occurrence_count > 0,
            attributes=attributes,
            child_paths=tuple(_display_path(key, namespaces) for key in stats.child_keys),
            likely_identifier_attributes=likely_identifiers,
            record_candidate_score=score,
            record_candidate_reasons=reasons,
        )
        profiles.append(profile)

    candidates = tuple(
        XmlRecordCandidate(
            path=profile.path,
            canonical_path=profile.canonical_path,
            score=profile.record_candidate_score,
            occurrence_count=profile.occurrence_count,
            depth=profile.depth,
            reasons=profile.record_candidate_reasons,
        )
        for profile in sorted(
            profiles,
            key=lambda item: (-item.record_candidate_score, -item.occurrence_count, item.depth, item.path),
        )
        if profile.record_candidate_score >= _RECORD_CANDIDATE_THRESHOLD
    )

    root_uri, root_local = _split_qname(root_tag)
    return XmlStructureArtifact(
        source_path=str(source),
        source_sha256=source_sha256,
        source_fingerprint=f"sha256:{source_sha256}",
        encoding=encoding,
        has_xml_declaration=has_declaration,
        root_tag=root_tag,
        root_local_name=root_local,
        root_namespace_uri=root_uri,
        root_prefix=_preferred_prefix(namespaces, root_uri),
        root_attribute_names=root_attributes,
        namespaces=tuple(namespaces),
        top_level_element_tags=tuple(top_level_tags),
        top_level_child_count=top_level_child_count,
        element_count=element_count,
        observed_max_depth=observed_max_depth,
        unique_element_path_count=len(profiles),
        leaf_path_count=sum(1 for item in profiles if not item.child_paths),
        repeated_path_count=sum(1 for item in profiles if item.repeated),
        optional_path_count=sum(1 for item in profiles if item.optional),
        mixed_content_path_count=sum(1 for item in profiles if item.mixed_content),
        element_profiles=tuple(profiles),
        record_candidates=candidates,
    )


def analyze_xml_structure(path: str | Path, **kwargs) -> XmlStructureArtifact:
    """Explicit Sprint-2 entry point for XML hierarchy/structure analysis."""
    return inspect_xml_structure(path, **kwargs)


def load_xml_structure(path: str | Path, **kwargs) -> XmlStructureArtifact:
    """Compatibility alias for callers that load the XML structural artifact directly."""
    return inspect_xml_structure(path, **kwargs)
