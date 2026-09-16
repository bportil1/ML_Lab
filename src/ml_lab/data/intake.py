from __future__ import annotations

import csv
import hashlib
import io
import os
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from .types import DataFileRecord, DataInventory, MalformedRow


_SUPPORTED: dict[str, str] = {
    ".csv": "delimited_text",
    ".tsv": "delimited_text",
}
_SAMPLE_BYTES = 128 * 1024
_INDEX_NAMES = {
    "index",
    "row",
    "row_index",
    "row_number",
    "rownum",
    "__index_level_0__",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_hidden(path: Path, root: Path) -> bool:
    try:
        parts = path.relative_to(root).parts
    except ValueError:
        parts = path.parts
    return any(part.startswith(".") and part not in {".", ".."} for part in parts)


def _discover_one(path: Path, *, recursive: bool, include_hidden: bool) -> Iterator[tuple[Path, Path]]:
    if path.is_file():
        yield path, path.parent
        return
    if not path.is_dir():
        return

    if recursive:
        for current, dirs, files in os.walk(path, followlinks=False):
            current_path = Path(current)
            if not include_hidden:
                dirs[:] = [name for name in dirs if not name.startswith(".")]
            for name in files:
                candidate = current_path / name
                if not include_hidden and _is_hidden(candidate, path):
                    continue
                if candidate.is_file():
                    yield candidate, path
    else:
        for candidate in sorted(path.iterdir()):
            if not candidate.is_file():
                continue
            if not include_hidden and candidate.name.startswith("."):
                continue
            yield candidate, path


def _detect_encoding(raw: bytes) -> tuple[str, list[str]]:
    warnings: list[str] = []
    if raw.startswith(b"\xef\xbb\xbf"):
        return "utf-8-sig", warnings
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return "utf-16", warnings
    for encoding in ("utf-8", "cp1252", "latin-1"):
        try:
            raw.decode(encoding)
            if encoding != "utf-8":
                warnings.append(f"encoding inferred as {encoding}; verify if text appears unusual")
            return encoding, warnings
        except UnicodeDecodeError:
            continue
    return "latin-1", ["encoding detection fell back to latin-1"]


def _dialect(sample: str, suffix: str) -> tuple[dict[str, object], bool, list[str]]:
    warnings: list[str] = []
    fallback = "\t" if suffix == ".tsv" else ","
    sniff = csv.Sniffer()
    try:
        dialect = sniff.sniff(sample, delimiters=",\t;|")
        kwargs: dict[str, object] = {
            "delimiter": dialect.delimiter,
            "quotechar": dialect.quotechar or '"',
            "doublequote": dialect.doublequote,
            "skipinitialspace": dialect.skipinitialspace,
            "quoting": dialect.quoting,
        }
        if dialect.escapechar:
            kwargs["escapechar"] = dialect.escapechar
    except csv.Error:
        kwargs = {"delimiter": fallback, "quotechar": '"'}
        warnings.append("delimiter sniffing failed; extension-based delimiter was used")

    header_sample = sample
    try:
        rows = [row for row in csv.reader(io.StringIO(sample), **kwargs) if row]
        if rows:
            modal_width = Counter(len(row) for row in rows).most_common(1)[0][0]
            regular = [row for row in rows if len(row) == modal_width][:25]
            if len(regular) >= 2:
                buffer = io.StringIO()
                writer = csv.writer(buffer, delimiter=str(kwargs["delimiter"]), lineterminator="\n")
                writer.writerows(regular)
                header_sample = buffer.getvalue()
    except (csv.Error, TypeError):
        pass

    try:
        has_header = sniff.has_header(header_sample)
    except csv.Error:
        has_header = True
        warnings.append("header detection was inconclusive; first row was treated as a header")
    return kwargs, has_header, warnings


def _display_columns(raw_columns: list[str]) -> tuple[list[str], list[str]]:
    warnings: list[str] = []
    names: list[str] = []
    seen: dict[str, int] = {}
    for index, value in enumerate(raw_columns, start=1):
        base = value.strip() or f"column_{index}"
        if not value.strip():
            warnings.append(f"blank header at column {index} was named {base}")
        count = seen.get(base, 0)
        seen[base] = count + 1
        if count:
            name = f"{base}.{count}"
            warnings.append(f"duplicate header {base!r} was disambiguated as {name!r}")
        else:
            name = base
        names.append(name)
    return names, warnings


def _int_sequence(values: list[str]) -> bool:
    if len(values) < 2:
        return False
    parsed: list[int] = []
    try:
        for value in values:
            stripped = value.strip()
            if stripped == "":
                return False
            number = float(stripped)
            if not number.is_integer():
                return False
            parsed.append(int(number))
    except ValueError:
        return False
    return parsed == list(range(parsed[0], parsed[0] + len(parsed))) and parsed[0] in {0, 1}


def _exported_indexes(raw_headers: list[str], displayed_headers: list[str], sample_rows: list[list[str]]) -> list[str]:
    indexes: list[str] = []
    if not sample_rows:
        return indexes
    for column_index, (raw_name, display_name) in enumerate(zip(raw_headers, displayed_headers)):
        normalized = raw_name.strip().lower()
        candidate_name = (
            normalized in _INDEX_NAMES
            or normalized.startswith("unnamed:")
            or (column_index == 0 and normalized == "")
        )
        if not candidate_name:
            continue
        values = [row[column_index] for row in sample_rows if len(row) > column_index]
        if _int_sequence(values):
            indexes.append(display_name)
    return indexes


def _inspect_delimited(path: Path, root: Path, *, preview_rows: int) -> DataFileRecord:
    size = path.stat().st_size
    digest = _sha256(path)
    suffix = path.suffix.lower()
    warnings: list[str] = []
    malformed: list[MalformedRow] = []
    malformed_count = 0

    with path.open("rb") as handle:
        raw = handle.read(_SAMPLE_BYTES)
    encoding, encoding_warnings = _detect_encoding(raw)
    warnings.extend(encoding_warnings)

    if b"\x00" in raw and encoding != "utf-16":
        return DataFileRecord(
            path=str(path.resolve()),
            relative_path=str(path.relative_to(root)),
            size_bytes=size,
            sha256=digest,
            suffix=suffix,
            format="delimited_text",
            supported=True,
            parse_status="failed",
            encoding=encoding,
            warnings=tuple(warnings),
            error="NUL bytes detected; file does not appear to be ordinary delimited text",
        )

    sample = raw.decode(encoding, errors="replace")
    dialect_kwargs, has_header, dialect_warnings = _dialect(sample, suffix)
    warnings.extend(dialect_warnings)
    delimiter = str(dialect_kwargs["delimiter"])

    raw_headers: list[str] = []
    columns: list[str] = []
    expected_width: int | None = None
    sample_rows: list[list[str]] = []
    row_count = 0
    parse_status = "parsed"
    parse_error: str | None = None

    try:
        with path.open("r", encoding=encoding, newline="", errors="strict") as handle:
            reader = csv.reader(handle, strict=True, **dialect_kwargs)
            try:
                first = next(reader)
            except StopIteration:
                first = []
            if first:
                if has_header:
                    raw_headers = list(first)
                    columns, header_warnings = _display_columns(raw_headers)
                    warnings.extend(header_warnings)
                    expected_width = len(first)
                else:
                    expected_width = len(first)
                    raw_headers = [f"column_{index}" for index in range(1, expected_width + 1)]
                    columns = list(raw_headers)
                    row_count = 1
                    sample_rows.append(list(first))

                for row in reader:
                    row_count += 1
                    if len(sample_rows) < max(2, preview_rows):
                        sample_rows.append(list(row))
                    if expected_width is not None and len(row) != expected_width:
                        malformed_count += 1
                        if len(malformed) < 20:
                            malformed.append(
                                MalformedRow(
                                    line_number=reader.line_num,
                                    observed_columns=len(row),
                                    expected_columns=expected_width,
                                    reason="column_count_mismatch",
                                )
                            )
            else:
                warnings.append("file is empty")
                raw_headers = []
                columns = []
                expected_width = 0
    except (UnicodeError, csv.Error) as exc:
        parse_status = "partial" if row_count or columns else "failed"
        parse_error = f"{type(exc).__name__}: {exc}"
        malformed_count += 1
        malformed.append(
            MalformedRow(
                line_number=0,
                observed_columns=None,
                expected_columns=expected_width,
                reason=parse_error,
            )
        )

    if malformed_count and parse_status == "parsed":
        parse_status = "partial"
        warnings.append(f"detected {malformed_count} malformed-width row(s)")

    exported = _exported_indexes(raw_headers, columns, sample_rows) if has_header else []
    if exported:
        warnings.append("likely exported index column(s): " + ", ".join(exported))

    return DataFileRecord(
        path=str(path.resolve()),
        relative_path=str(path.relative_to(root)),
        size_bytes=size,
        sha256=digest,
        suffix=suffix,
        format="delimited_text",
        supported=True,
        parse_status=parse_status,
        encoding=encoding,
        delimiter=delimiter,
        has_header=has_header,
        row_count=row_count,
        column_count=len(columns),
        columns=tuple(columns),
        likely_exported_index_columns=tuple(exported),
        malformed_row_count=malformed_count,
        malformed_rows=tuple(malformed),
        warnings=tuple(warnings),
        error=parse_error,
    )


def _unsupported(path: Path, root: Path) -> DataFileRecord:
    size = path.stat().st_size
    return DataFileRecord(
        path=str(path.resolve()),
        relative_path=str(path.relative_to(root)),
        size_bytes=size,
        sha256="",
        suffix=path.suffix.lower(),
        format="unknown",
        supported=False,
        parse_status="unsupported",
        warnings=("format is not supported by Data Lab A1",),
    )


def inspect_file(path: str | Path, *, root: str | Path | None = None, preview_rows: int = 20) -> DataFileRecord:
    candidate = Path(path).expanduser()
    if not candidate.is_file():
        raise FileNotFoundError(f"not a file: {candidate}")
    base = Path(root).expanduser() if root is not None else candidate.parent
    suffix = candidate.suffix.lower()
    if suffix in _SUPPORTED:
        return _inspect_delimited(candidate, base, preview_rows=preview_rows)
    return _unsupported(candidate, base)


def inspect_paths(
    paths: Iterable[str | Path],
    *,
    recursive: bool = True,
    include_hidden: bool = False,
    preview_rows: int = 20,
) -> DataInventory:
    requested = tuple(str(Path(path).expanduser()) for path in paths)
    if not requested:
        raise ValueError("at least one file or directory path is required")
    if preview_rows < 1:
        raise ValueError("preview_rows must be at least 1")

    warnings: list[str] = []
    discovered: dict[str, tuple[Path, Path]] = {}
    for raw_path in requested:
        candidate = Path(raw_path)
        if not candidate.exists():
            warnings.append(f"path does not exist: {candidate}")
            continue
        for file_path, root in _discover_one(candidate, recursive=recursive, include_hidden=include_hidden):
            discovered[str(file_path.resolve())] = (file_path, root)

    records: list[DataFileRecord] = []
    for _, (file_path, root) in sorted(discovered.items(), key=lambda item: item[0]):
        try:
            records.append(inspect_file(file_path, root=root, preview_rows=preview_rows))
        except (OSError, UnicodeError, csv.Error) as exc:
            try:
                size = file_path.stat().st_size
            except OSError:
                size = 0
            records.append(
                DataFileRecord(
                    path=str(file_path.resolve()),
                    relative_path=str(file_path.relative_to(root)),
                    size_bytes=size,
                    sha256="",
                    suffix=file_path.suffix.lower(),
                    format=_SUPPORTED.get(file_path.suffix.lower(), "unknown"),
                    supported=file_path.suffix.lower() in _SUPPORTED,
                    parse_status="failed",
                    error=f"{type(exc).__name__}: {exc}",
                )
            )

    supported = [record for record in records if record.supported]
    return DataInventory(
        requested_paths=requested,
        recursive=recursive,
        include_hidden=include_hidden,
        discovered_file_count=len(records),
        supported_file_count=len(supported),
        parsed_file_count=sum(record.parse_status == "parsed" for record in records),
        partial_file_count=sum(record.parse_status == "partial" for record in records),
        failed_file_count=sum(record.parse_status == "failed" for record in records),
        unsupported_file_count=sum(not record.supported for record in records),
        total_size_bytes=sum(record.size_bytes for record in records),
        files=tuple(records),
        warnings=tuple(warnings),
    )
