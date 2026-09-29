from __future__ import annotations

from .types import XmlElementProfile, XmlFieldCandidate, XmlRecordSelection, XmlStructureArtifact


class XmlSelectionError(ValueError):
    """Raised when an XML record-root/field-selection request is invalid."""


def _profile_map(artifact: XmlStructureArtifact) -> dict[str, XmlElementProfile]:
    return {item.canonical_path: item for item in artifact.element_profiles}


def _is_descendant_or_self(path: str, root: str) -> bool:
    return path == root or path.startswith(root + "/")


def _relative_display_path(path: str, root_path: str) -> str:
    if path == root_path:
        return "."
    prefix = root_path.rstrip("/") + "/"
    return path[len(prefix):] if path.startswith(prefix) else path


def _ancestor_chain(profile_by_path: dict[str, XmlElementProfile], canonical_path: str, root: str) -> list[XmlElementProfile]:
    chain: list[XmlElementProfile] = []
    cursor = canonical_path
    while cursor and _is_descendant_or_self(cursor, root):
        profile = profile_by_path.get(cursor)
        if profile is not None:
            chain.append(profile)
        if cursor == root:
            break
        cursor = cursor.rsplit("/", 1)[0]
    return list(reversed(chain))


def _branch_flags(profile_by_path: dict[str, XmlElementProfile], canonical_path: str, root: str) -> tuple[bool, bool]:
    chain = _ancestor_chain(profile_by_path, canonical_path, root)
    # The chosen record root itself defines one record. Multiplicity/optionality
    # matters only below that boundary.
    descendants = chain[1:] if chain and chain[0].canonical_path == root else chain
    return any(item.repeated for item in descendants), any(item.optional for item in descendants)


def build_xml_record_selection(
    artifact: XmlStructureArtifact,
    record_root_canonical_path: str,
) -> XmlRecordSelection:
    """Describe selectable scalar fields beneath one explicit XML record root.

    This is intentionally selection-only. It does not read values, flatten XML,
    or produce a table; those operations belong to later XML sprints.
    """

    root_key = (record_root_canonical_path or "").strip()
    if not root_key:
        raise XmlSelectionError("record root canonical path is required")

    profile_by_path = _profile_map(artifact)
    root = profile_by_path.get(root_key)
    if root is None:
        raise XmlSelectionError(f"record root path is not present in this XML structure: {root_key}")

    fields: list[XmlFieldCandidate] = []
    for profile in artifact.element_profiles:
        if not _is_descendant_or_self(profile.canonical_path, root_key):
            continue

        branch_repeated, branch_optional = _branch_flags(profile_by_path, profile.canonical_path, root_key)
        relative_element = _relative_display_path(profile.path, root.path)

        # Mixed-content text is not a safe scalar because flattening only the leading
        # text would silently discard interleaved child/tail content. Attributes and
        # descendant leaf text remain selectable.
        if profile.text_occurrence_count > 0 and not profile.mixed_content:
            field_path = profile.path
            relative_path = relative_element
            fields.append(
                XmlFieldCandidate(
                    field_id=f"text:{profile.canonical_path}",
                    kind="element_text",
                    path=field_path,
                    canonical_path=profile.canonical_path,
                    relative_path=relative_path,
                    source_element_path=profile.path,
                    source_element_canonical_path=profile.canonical_path,
                    name=profile.local_name,
                    local_name=profile.local_name,
                    namespace_uri=profile.namespace_uri,
                    prefix=profile.prefix,
                    repeated=branch_repeated,
                    optional=branch_optional or profile.text_presence_rate < 1.0,
                    likely_identifier=profile.likely_identifier_text,
                )
            )

        for attribute in profile.attributes:
            display_name = f"{attribute.prefix}:{attribute.local_name}" if attribute.prefix else attribute.local_name
            relative_attr = f"@{display_name}" if relative_element == "." else f"{relative_element}/@{display_name}"
            fields.append(
                XmlFieldCandidate(
                    field_id=f"attribute:{profile.canonical_path}@{attribute.name}",
                    kind="attribute",
                    path=f"{profile.path}/@{display_name}",
                    canonical_path=f"{profile.canonical_path}/@{attribute.name}",
                    relative_path=relative_attr,
                    source_element_path=profile.path,
                    source_element_canonical_path=profile.canonical_path,
                    name=display_name,
                    local_name=attribute.local_name,
                    namespace_uri=attribute.namespace_uri,
                    prefix=attribute.prefix,
                    repeated=branch_repeated,
                    optional=branch_optional or attribute.presence_rate < 1.0,
                    likely_identifier=attribute.likely_identifier,
                )
            )

    fields.sort(key=lambda item: (item.relative_path.count("/"), item.relative_path, item.kind))
    return XmlRecordSelection(
        source_fingerprint=artifact.source_fingerprint,
        record_root_path=root.path,
        record_root_canonical_path=root.canonical_path,
        record_root_occurrence_count=root.occurrence_count,
        record_root_candidate_score=root.record_candidate_score,
        record_root_candidate_reasons=root.record_candidate_reasons,
        fields=tuple(fields),
    )
