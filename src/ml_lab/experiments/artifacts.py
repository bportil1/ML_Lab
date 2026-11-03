"""Project-local artifact references and run-linked provenance.

The registry indexes existing project files; it does not copy, rewrite, or
execute them. It is independent of optional scientific and UI dependencies.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import stat
from typing import Any, Mapping, Sequence, TYPE_CHECKING
from uuid import uuid4

from .store import ExperimentNotFound, ExperimentStoreError, _json_object, _nonempty_string, _pagination, _utcnow

if TYPE_CHECKING:
    from .store import ExperimentStore

_KIND_PATTERN = re.compile(r"^[a-z][a-z0-9_-]{0,63}$")
_ROLES = frozenset({"input", "output"})


class ArtifactError(ExperimentStoreError):
    """Invalid artifact path, metadata, or provenance reference."""


class ArtifactNotFound(ExperimentNotFound):
    """Artifact identifier is absent from this project's registry."""


def _hash_file(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ArtifactError("artifact must be an existing regular file")
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
            size += len(block)
        after = os.fstat(stream.fileno())
    if ((before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
            != (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            or size != after.st_size):
        raise ArtifactError("artifact changed during hashing; retry after writing completes")
    return size, digest.hexdigest()


class ArtifactRegistry:
    """Typed references to files produced or consumed by experiment runs.

    Paths are canonical, project-relative and must point to existing regular
    files *inside* the owning project. Symlinked path components are rejected.
    Provenance edges reference artifacts already registered in this project.
    """

    def __init__(self, store: ExperimentStore) -> None:
        self.store = store

    def _project_file(self, path: str | Path) -> tuple[Path, str]:
        if not isinstance(path, (str, Path)) or not str(path).strip():
            raise ArtifactError("path must be a nonempty filesystem path")
        candidate = Path(path).expanduser()
        if ".." in candidate.parts:
            raise ArtifactError("path traversal is not permitted")
        if not candidate.is_absolute():
            candidate = self.store.project_root / candidate
        try:
            relative = candidate.relative_to(self.store.project_root)
            # Refuse symlinks, even if they currently point back inside the
            # project. This prevents accidentally indexing another project.
            current = self.store.project_root
            for part in relative.parts:
                current = current / part
                if current.is_symlink():
                    raise ArtifactError("symlinked artifact paths are not supported")
            resolved = candidate.resolve(strict=True)
            relative = resolved.relative_to(self.store.project_root)
            if not resolved.is_file():
                raise ArtifactError("artifact must be an existing regular file")
        except (OSError, ValueError, RuntimeError) as exc:
            raise ArtifactError("artifact must be an existing file inside this project") from exc
        return resolved, relative.as_posix()

    @staticmethod
    def _artifact_from_row(db: sqlite3.Connection, row: sqlite3.Row) -> dict[str, Any]:
        sources = db.execute(
            "SELECT source_artifact_id FROM artifact_sources WHERE artifact_id = ? ORDER BY position",
            (row["artifact_id"],),
        ).fetchall()
        return {
            "schema": "ml-lab.artifact@1",
            "artifact_id": row["artifact_id"],
            "run_id": row["run_id"],
            "experiment_id": row["experiment_id"],
            "revision": row["revision"],
            "role": row["role"],
            "kind": row["kind"],
            "name": row["name"],
            "path": row["relative_path"],
            "media_type": row["media_type"],
            "size_bytes": row["size_bytes"],
            "sha256": row["sha256"],
            "created_at": row["created_at"],
            "metadata": json.loads(row["metadata_json"]),
            "source_artifact_ids": [item["source_artifact_id"] for item in sources],
        }

    def register(
        self,
        run_id: str,
        path: str | Path,
        *,
        name: str | None = None,
        role: str = "output",
        kind: str = "other",
        media_type: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        source_artifact_ids: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Index an existing file, its content fingerprint, and direct sources.

        A run can have several artifacts, and an artifact may refer to sources
        from earlier runs in the same project. Registration never mutates files
        or changes execution status; the producing executor reports run state.
        """
        run_id = _nonempty_string(run_id, "run_id")
        if role not in _ROLES:
            raise ArtifactError(f"role must be one of: {', '.join(sorted(_ROLES))}")
        if not isinstance(kind, str) or not _KIND_PATTERN.fullmatch(kind):
            raise ArtifactError("kind must be a lowercase identifier (up to 64 characters)")
        if media_type is not None:
            media_type = _nonempty_string(media_type, "media_type")
        metadata_json = _json_object(metadata, "metadata")
        if isinstance(source_artifact_ids, (str, bytes)) or not isinstance(source_artifact_ids, Sequence):
            raise ArtifactError("source_artifact_ids must be a sequence of artifact IDs")
        sources = tuple(_nonempty_string(source, "source artifact ID") for source in source_artifact_ids)
        if len(sources) > 500 or len(set(sources)) != len(sources):
            raise ArtifactError("source_artifact_ids must contain at most 500 unique IDs")

        file_path, relative_path = self._project_file(path)
        artifact_name = _nonempty_string(name if name is not None else file_path.name, "name")
        try:
            size, sha256 = _hash_file(file_path)
        except OSError as exc:
            raise ArtifactError("could not read artifact file") from exc
        artifact_id = uuid4().hex
        with self.store._connection(write=True) as db:
            if db.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is None:
                raise ExperimentNotFound(f"Unknown run: {run_id}")
            if sources:
                placeholders = ",".join("?" for _ in sources)
                found = db.execute(
                    f"SELECT artifact_id FROM artifacts WHERE artifact_id IN ({placeholders})", sources
                ).fetchall()
                if len(found) != len(sources):
                    raise ArtifactNotFound("source artifact is not registered in this project")
            db.execute(
                """INSERT INTO artifacts
                   (artifact_id, run_id, role, kind, name, relative_path, media_type,
                    size_bytes, sha256, created_at, metadata_json)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (artifact_id, run_id, role, kind, artifact_name, relative_path,
                 media_type, size, sha256, _utcnow(), metadata_json),
            )
            for position, source in enumerate(sources):
                db.execute(
                    "INSERT INTO artifact_sources (artifact_id, source_artifact_id, position) VALUES (?, ?, ?)",
                    (artifact_id, source, position),
                )
        return self.get(artifact_id)

    def get(self, artifact_id: str) -> dict[str, Any]:
        with self.store._connection() as db:
            row = db.execute(
                """SELECT a.*, r.experiment_id, r.revision FROM artifacts a
                   JOIN runs r ON a.run_id = r.run_id WHERE a.artifact_id = ?""",
                (artifact_id,),
            ).fetchone()
            if row is None:
                raise ArtifactNotFound(f"Unknown artifact: {artifact_id}")
            return self._artifact_from_row(db, row)

    def list_for_run(self, run_id: str, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        limit, offset = _pagination(limit, offset)
        with self.store._connection() as db:
            if db.execute("SELECT 1 FROM runs WHERE run_id = ?", (run_id,)).fetchone() is None:
                raise ExperimentNotFound(f"Unknown run: {run_id}")
            rows = db.execute(
                """SELECT a.*, r.experiment_id, r.revision FROM artifacts a
                   JOIN runs r ON a.run_id = r.run_id
                   WHERE a.run_id = ? ORDER BY a.created_at DESC, a.artifact_id DESC
                   LIMIT ? OFFSET ?""",
                (run_id, limit, offset),
            ).fetchall()
            return [self._artifact_from_row(db, row) for row in rows]

    def list_for_experiment(
        self, experiment_id: str, *, limit: int = 100, offset: int = 0,
    ) -> list[dict[str, Any]]:
        limit, offset = _pagination(limit, offset)
        with self.store._connection() as db:
            if db.execute("SELECT 1 FROM experiments WHERE experiment_id = ?", (experiment_id,)).fetchone() is None:
                raise ExperimentNotFound(f"Unknown experiment: {experiment_id}")
            rows = db.execute(
                """SELECT a.*, r.experiment_id, r.revision FROM artifacts a
                   JOIN runs r ON a.run_id = r.run_id
                   WHERE r.experiment_id = ? ORDER BY a.created_at DESC, a.artifact_id DESC
                   LIMIT ? OFFSET ?""",
                (experiment_id, limit, offset),
            ).fetchall()
            return [self._artifact_from_row(db, row) for row in rows]

    def verify(self, artifact_id: str) -> dict[str, Any]:
        """Rehash a registered file; never silently update its recorded digest."""
        record = self.get(artifact_id)
        result: dict[str, Any] = {"artifact_id": artifact_id, "status": "missing"}
        try:
            resolved, rel = self._project_file(record["path"])
            if rel != record["path"]:
                return {**result, "status": "unsafe"}
            size, sha256 = _hash_file(resolved)
        except ArtifactError:
            # A missing file and a path that now escapes the project should
            # not be silently confused with a valid registration.
            candidate = self.store.project_root / record["path"]
            return {**result, "status": "missing" if not candidate.exists() and not candidate.is_symlink() else "unsafe"}
        except OSError:
            return {**result, "status": "unreadable"}
        return {
            **result,
            "status": "valid" if (size == record["size_bytes"] and sha256 == record["sha256"]) else "changed",
            "size_bytes": size,
            "sha256": sha256,
        }
