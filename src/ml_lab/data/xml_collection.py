"""XML-C1: local, read-only, bounded directory inventory for XML collections.

This is discovery, not similarity assessment, schema unification, or CSV extraction.
Individual sources retain their own XML parser safety constraints and fingerprints.
"""

from __future__ import annotations

import fnmatch
import json
import os
import stat
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .xml_ingest import XmlSafetyError, inspect_xml_structure


@dataclass(frozen=True)
class XmlCollectionSource:
    path: str
    relative_path: str
    size_bytes: int
    status: str
    sha256: str = ""
    encoding: str | None = None
    root_tag: str | None = None
    root_local_name: str | None = None
    root_namespace_uri: str | None = None
    namespaces: tuple[tuple[str, str], ...] = ()
    element_count: int | None = None
    unique_path_count: int | None = None
    record_candidates: tuple[dict[str, Any], ...] = ()
    error: str | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "relative_path": self.relative_path,
            "size_bytes": self.size_bytes,
            "status": self.status,
            "sha256": self.sha256,
            "source_fingerprint": f"sha256:{self.sha256}" if self.sha256 else None,
            "encoding": self.encoding,
            "root": {
                "tag": self.root_tag,
                "local_name": self.root_local_name,
                "namespace_uri": self.root_namespace_uri,
            } if self.root_tag is not None else None,
            "namespaces": [{"prefix": prefix, "uri": uri} for prefix, uri in self.namespaces],
            "element_count": self.element_count,
            "unique_path_count": self.unique_path_count,
            "record_candidates": list(self.record_candidates),
            "error": self.error,
        }


@dataclass(frozen=True)
class XmlCollectionInventory:
    root_path: str
    recursive: bool
    include_hidden: bool
    include_globs: tuple[str, ...]
    exclude_globs: tuple[str, ...]
    files: tuple[XmlCollectionSource, ...]
    total_size_bytes: int
    schema: str = "ml-lab.xml-collection-inventory@1"

    def to_record(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "root_path": self.root_path,
            "settings": {
                "recursive": self.recursive,
                "include_hidden": self.include_hidden,
                "include_globs": list(self.include_globs),
                "exclude_globs": list(self.exclude_globs),
            },
            "summary": {
                "file_count": len(self.files),
                "parsed_file_count": sum(item.status == "parsed" for item in self.files),
                "failed_file_count": sum(item.status == "failed" for item in self.files),
                "total_size_bytes": self.total_size_bytes,
                "similarity_status": "not_computed",
                "consolidation_status": "not_available",
            },
            "files": [item.to_record() for item in self.files],
        }


def _matches(relative: str, patterns: tuple[str, ...]) -> bool:
    """Match case-insensitively against a basename or a root-relative POSIX path."""
    name = relative.rsplit("/", 1)[-1].casefold()
    lowered = relative.casefold()
    return any(fnmatch.fnmatchcase(name, pattern.casefold()) or
               fnmatch.fnmatchcase(lowered, pattern.casefold()) for pattern in patterns)


def inventory_xml_collection(
    directory: str | Path,
    *,
    recursive: bool = True,
    include_hidden: bool = False,
    include_globs: Iterable[str] = ("*.xml",),
    exclude_globs: Iterable[str] = (),
    max_files: int = 1000,
    max_entries: int = 100_000,
    max_total_bytes: int = 2 * 1024 ** 3,
    max_file_bytes: int = 512 * 1024 ** 2,
) -> XmlCollectionInventory:
    """Inventory XML sources; fail closed when discovery limits are exceeded.

    Symlinked source files/directories are not followed. Malformed, unsupported,
    or unreadable individual XML files are reported instead of hiding siblings.
    No schema similarity, dataset joining, or persistence is implied by this call.
    """
    source_root = Path(directory).expanduser()
    if source_root.is_symlink() or not source_root.is_dir():
        raise ValueError(f"expected a real, accessible directory: {source_root}")
    root = source_root.resolve()
    includes = tuple(str(item).strip() for item in include_globs if str(item).strip())
    excludes = tuple(str(item).strip() for item in exclude_globs if str(item).strip())
    if not includes:
        raise ValueError("at least one include glob is required")
    if any(value < 1 for value in (max_files, max_entries, max_total_bytes, max_file_bytes)):
        raise ValueError("all collection limits must be positive")

    discovered: list[tuple[Path, str]] = []
    examined = 0
    directory_count = 0

    def fail_on_limit() -> None:
        if len(discovered) > max_files:
            raise ValueError(f"XML collection file limit exceeded ({max_files})")

    def on_error(exc: OSError) -> None:
        raise exc

    for current, dirs, names in os.walk(root, topdown=True, followlinks=False, onerror=on_error):
        directory_count += 1
        if directory_count > max_entries:
            raise ValueError(f"XML collection discovery directory limit exceeded ({max_entries})")
        current_path = Path(current)
        dirs[:] = sorted((name for name in dirs
                          if (include_hidden or not name.startswith("."))
                          and not (current_path / name).is_symlink())) if recursive else []
        for name in sorted(names):
            examined += 1
            if examined > max_entries:
                raise ValueError(f"XML collection discovery entry limit exceeded ({max_entries})")
            if not include_hidden and name.startswith("."):
                continue
            candidate = current_path / name
            if candidate.is_symlink():
                continue
            relative = candidate.relative_to(root).as_posix()
            if candidate.suffix.casefold() != ".xml":
                continue
            if not _matches(relative, includes) or _matches(relative, excludes):
                continue
            discovered.append((candidate, relative))
            fail_on_limit()
        if not recursive:
            break

    discovered.sort(key=lambda item: item[1])
    result: list[XmlCollectionSource] = []
    total_bytes = 0
    for candidate, relative in discovered:
        absolute = str(candidate)
        if candidate.is_symlink():
            result.append(XmlCollectionSource(absolute, relative, 0, "failed", error="source became a symlink during inventory"))
            continue
        try:
            info = candidate.lstat()
            if not stat.S_ISREG(info.st_mode):
                result.append(XmlCollectionSource(absolute, relative, 0, "failed", error="not a regular XML file"))
                continue
            if not candidate.resolve().is_relative_to(root):
                result.append(XmlCollectionSource(absolute, relative, 0, "failed", error="source escapes selected directory"))
                continue
            size = info.st_size
        except OSError as exc:
            result.append(XmlCollectionSource(absolute, relative, 0, "failed", error=f"{type(exc).__name__}: {exc}"))
            continue
        total_bytes += size
        if total_bytes > max_total_bytes:
            raise ValueError(f"XML collection total size limit exceeded ({max_total_bytes:,} bytes)")
        if size > max_file_bytes:
            result.append(XmlCollectionSource(absolute, relative, size, "failed",
                                              error=f"XML file size limit exceeded ({max_file_bytes:,} bytes)"))
            continue
        try:
            artifact = inspect_xml_structure(candidate, max_source_bytes=max_file_bytes)
            if candidate.stat().st_size != size:
                raise XmlSafetyError("XML source changed during inventory")
        except (OSError, XmlSafetyError, ET.ParseError, LookupError, UnicodeError, ValueError) as exc:
            result.append(XmlCollectionSource(absolute, relative, size, "failed", error=f"{type(exc).__name__}: {exc}"))
            continue
        result.append(XmlCollectionSource(
            path=absolute,
            relative_path=relative,
            size_bytes=size,
            status="parsed",
            sha256=artifact.source_sha256,
            encoding=artifact.encoding,
            root_tag=artifact.root_tag,
            root_local_name=artifact.root_local_name,
            root_namespace_uri=artifact.root_namespace_uri,
            namespaces=tuple((item.prefix, item.uri) for item in artifact.namespaces),
            element_count=artifact.element_count,
            unique_path_count=artifact.unique_element_path_count,
            record_candidates=tuple(item.to_record() for item in artifact.record_candidates[:10]),
        ))
    return XmlCollectionInventory(str(root), recursive, include_hidden, includes, excludes,
                                  tuple(result), total_bytes)


def save_xml_collection_inventory(inventory: XmlCollectionInventory, destination: str | Path) -> Path:
    """Explicitly persist inventory metadata; never rewrite source XML."""
    target = Path(destination).expanduser()
    if target.suffix.lower() != ".json":
        target /= "xml_collection_inventory.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(inventory.to_record(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target
