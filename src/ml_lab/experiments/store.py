"""Per-project, versioned SQLite experiment metadata store.

S1.1 is an opt-in storage API, not a new task dispatcher or training runner.
No PAH, Flask, pandas, sklearn, or torch dependency is required. The selected
project root is explicit, or defaults to the standalone process's working
folder at construction time. It never uses a shared user-wide catalog.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
from typing import Any, Iterator, Mapping
from uuid import uuid4

SCHEMA_VERSION = 3
_EXECUTION_STATUSES = frozenset({"queued", "running", "succeeded", "failed", "cancelled"})
_TERMINAL = frozenset({"succeeded", "failed", "cancelled"})
_TRANSITIONS = {
    "queued": frozenset({"running", "succeeded", "failed", "cancelled"}),
    "running": _TERMINAL,
    "succeeded": frozenset(),
    "failed": frozenset(),
    "cancelled": frozenset(),
}


class ExperimentStoreError(ValueError):
    """Invalid persisted experiment metadata or store operation."""


class SchemaVersionError(ExperimentStoreError):
    """The on-disk schema is newer than this ML_Lab implementation."""


class ExperimentNotFound(ExperimentStoreError):
    """The requested experiment or run does not exist in this project."""


class InvalidRunTransition(ExperimentStoreError):
    """A run cannot move between the requested execution statuses."""


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_object(value: Mapping[str, Any] | None, field: str) -> str:
    if value is None:
        return "{}"
    if not isinstance(value, Mapping):
        raise ExperimentStoreError(f"{field} must be a JSON object or None")
    try:
        encoded = json.dumps(dict(value), sort_keys=True, separators=(",", ":"), allow_nan=False)
        decoded = json.loads(encoded)
    except (TypeError, ValueError) as exc:
        raise ExperimentStoreError(f"{field} must contain JSON-compatible values") from exc
    if not isinstance(decoded, dict):
        raise ExperimentStoreError(f"{field} must be a JSON object")
    return encoded


def _nonempty_string(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ExperimentStoreError(f"{field} must be a nonempty string")
    return value.strip()


def _pagination(limit: int, offset: int) -> tuple[int, int]:
    if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 500:
        raise ExperimentStoreError("limit must be between 1 and 500")
    if not isinstance(offset, int) or isinstance(offset, bool) or offset < 0:
        raise ExperimentStoreError("offset must be nonnegative")
    return limit, offset


class ExperimentStore:
    """Store experiment revisions and run metadata within one project.

    Instantiate with ``project_root`` for PAH-selected projects; standalone
    callers can omit it to use the current working directory. Only this
    instance's ``analysis/ml_lab/experiments.sqlite3`` is accessed. Large
    datasets, weights, and artifact manifests remain on disk and are *not*
    embedded as SQLite blobs. Artifacts are registered separately through
    the ``artifacts`` property.
    """

    def __init__(self, project_root: str | Path | None = None) -> None:
        self.project_root = Path(project_root if project_root is not None else Path.cwd()).expanduser().resolve()
        if not self.project_root.is_dir():
            raise ExperimentStoreError(f"Project directory does not exist: {self.project_root}")
        self.db_path = self.project_root / "analysis" / "ml_lab" / "experiments.sqlite3"
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def _connection(self, *, write: bool = False) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        connection.execute("PRAGMA foreign_keys = ON")
        try:
            if write:
                connection.execute("BEGIN IMMEDIATE")
            yield connection
            if write:
                connection.execute("COMMIT")
        except BaseException:
            if write and connection.in_transaction:
                connection.execute("ROLLBACK")
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        # WAL permits concurrent readers alongside atomic, serialized writers.
        with self._connection() as db:
            db.execute("PRAGMA journal_mode = WAL")
        with self._connection(write=True) as db:
            version = int(db.execute("PRAGMA user_version").fetchone()[0])
            if version > SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"Database schema {version} is newer than supported version {SCHEMA_VERSION}"
                )
            if version == 0:
                # Version 0 is reserved for a new, empty store: never overwrite
                # unversioned tables that may belong to another application.
                existing = db.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
                ).fetchall()
                if existing:
                    raise SchemaVersionError("Unversioned database is not empty; refusing to initialize")
                # Individual statements preserve the surrounding BEGIN IMMEDIATE;
                # executescript() would commit the transaction prematurely.
                for statement in (
                    """CREATE TABLE experiments (
                        experiment_id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        current_revision INTEGER NOT NULL CHECK(current_revision >= 1)
                    )""",
                    """CREATE TABLE experiment_revisions (
                        experiment_id TEXT NOT NULL REFERENCES experiments(experiment_id),
                        revision INTEGER NOT NULL CHECK(revision >= 1),
                        created_at TEXT NOT NULL,
                        dataset_ref_json TEXT NOT NULL,
                        model_spec_json TEXT NOT NULL,
                        evaluation_plan_json TEXT NOT NULL,
                        metadata_json TEXT NOT NULL,
                        PRIMARY KEY (experiment_id, revision)
                    )""",
                    """CREATE TABLE runs (
                        run_id TEXT PRIMARY KEY,
                        experiment_id TEXT NOT NULL,
                        revision INTEGER NOT NULL,
                        status TEXT NOT NULL CHECK(status IN ('queued','running','succeeded','failed','cancelled')),
                        created_at TEXT NOT NULL,
                        started_at TEXT,
                        finished_at TEXT,
                        seed INTEGER,
                        environment_json TEXT NOT NULL,
                        error TEXT,
                        FOREIGN KEY (experiment_id, revision)
                            REFERENCES experiment_revisions(experiment_id, revision)
                    )""",
                    "CREATE INDEX runs_by_experiment ON runs(experiment_id, created_at, run_id)",
                ):
                    db.execute(statement)
                db.execute("PRAGMA user_version = 1")
            if version in (1, 0):
                # A nondestructive v1 -> v2 migration. Previously recorded
                # experiments and runs keep the same IDs and snapshots.
                for statement in (
                    """CREATE TABLE artifacts (
                        artifact_id TEXT PRIMARY KEY,
                        run_id TEXT NOT NULL REFERENCES runs(run_id),
                        role TEXT NOT NULL CHECK(role IN ('input','output')),
                        kind TEXT NOT NULL,
                        name TEXT NOT NULL,
                        relative_path TEXT NOT NULL,
                        media_type TEXT,
                        size_bytes INTEGER NOT NULL CHECK(size_bytes >= 0),
                        sha256 TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        metadata_json TEXT NOT NULL
                    )""",
                    """CREATE TABLE artifact_sources (
                        artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
                        source_artifact_id TEXT NOT NULL REFERENCES artifacts(artifact_id),
                        position INTEGER NOT NULL CHECK(position >= 0),
                        PRIMARY KEY (artifact_id, source_artifact_id),
                        UNIQUE (artifact_id, position),
                        CHECK (artifact_id != source_artifact_id)
                    )""",
                    "CREATE INDEX artifacts_by_run ON artifacts(run_id, created_at, artifact_id)",
                    "CREATE INDEX artifact_sources_by_source ON artifact_sources(source_artifact_id)",
                ):
                    db.execute(statement)
                db.execute("PRAGMA user_version = 2")
            if version in (0, 1, 2):
                # v3 adds only a project-local active-workbench pointer. Dataset
                # bytes and experiment revisions remain in their existing stores.
                # UUIDs are intentionally independent of Flask/browser sessions
                # so multiple workbench instances can be supported in the future.
                db.execute(
                    """CREATE TABLE IF NOT EXISTS workbench_instances (
                        instance_id TEXT PRIMARY KEY,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        generation INTEGER NOT NULL DEFAULT 0 CHECK(generation >= 0),
                        experiment_id TEXT,
                        experiment_revision INTEGER,
                        CHECK ((experiment_id IS NULL AND experiment_revision IS NULL)
                            OR (experiment_id IS NOT NULL AND experiment_revision >= 1)),
                        FOREIGN KEY (experiment_id, experiment_revision)
                            REFERENCES experiment_revisions(experiment_id, revision)
                    )"""
                )
                db.execute("PRAGMA user_version = 3")
            if version in (0, 1, 2, 3):
                # Optional, v3-compatible table: old v3 versions ignore it,
                # making rollback non-destructive. It is kept in the SAME
                # project store and never changes source data or revisions.
                db.execute(
                    """CREATE TABLE IF NOT EXISTS workbench_feature_roles (
                        instance_id TEXT NOT NULL REFERENCES workbench_instances(instance_id)
                            ON DELETE CASCADE,
                        dataset_sha256 TEXT NOT NULL,
                        column_name TEXT NOT NULL,
                        role TEXT NOT NULL CHECK(role IN ('continuous','discrete','identifier','exclude')),
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (instance_id, dataset_sha256, column_name)
                    )"""
                )
            if version in (0, 1, 2, 3):
                # Optional local cache, not a new schema version. Older v3
                # binaries ignore this table if the patch is rolled back.
                db.execute(
                    """CREATE TABLE IF NOT EXISTS workbench_feature_profile_cache (
                        instance_id TEXT NOT NULL REFERENCES workbench_instances(instance_id)
                            ON DELETE CASCADE,
                        dataset_sha256 TEXT NOT NULL,
                        cache_key TEXT NOT NULL,
                        report_json TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        PRIMARY KEY (instance_id, dataset_sha256, cache_key)
                    )"""
                )
            # Refuse future versions and do not guess schemas.

    @property
    def artifacts(self) -> "ArtifactRegistry":
        """Get an independent project-local registry for file artifacts."""
        from .artifacts import ArtifactRegistry
        return ArtifactRegistry(self)

    @staticmethod
    def _experiment_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "schema": "ml-lab.experiment@1",
            "experiment_id": row["experiment_id"],
            "name": row["name"],
            "revision": row["revision"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "revision_created_at": row["revision_created_at"],
            "dataset_ref": json.loads(row["dataset_ref_json"]),
            "model_spec": json.loads(row["model_spec_json"]),
            "evaluation_plan": json.loads(row["evaluation_plan_json"]),
            "metadata": json.loads(row["metadata_json"]),
        }

    @staticmethod
    def _run_from_row(row: sqlite3.Row) -> dict[str, Any]:
        return {
            "schema": "ml-lab.experiment-run@1",
            "run_id": row["run_id"],
            "experiment_id": row["experiment_id"],
            "revision": row["revision"],
            "status": row["status"],
            "created_at": row["created_at"],
            "started_at": row["started_at"],
            "finished_at": row["finished_at"],
            "seed": row["seed"],
            "environment": json.loads(row["environment_json"]),
            "error": row["error"],
        }

    def create_experiment(
        self,
        name: str,
        *,
        dataset_ref: Mapping[str, Any] | None = None,
        model_spec: Mapping[str, Any] | None = None,
        evaluation_plan: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        name = _nonempty_string(name, "name")
        fields = tuple(_json_object(value, key) for key, value in (
            ("dataset_ref", dataset_ref), ("model_spec", model_spec),
            ("evaluation_plan", evaluation_plan), ("metadata", metadata),
        ))
        identifier = uuid4().hex
        now = _utcnow()
        with self._connection(write=True) as db:
            db.execute(
                "INSERT INTO experiments VALUES (?, ?, ?, ?, ?)",
                (identifier, name, now, now, 1),
            )
            db.execute(
                "INSERT INTO experiment_revisions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (identifier, 1, now, *fields),
            )
        return self.get_experiment(identifier)

    def revise_experiment(
        self,
        experiment_id: str,
        *,
        dataset_ref: Mapping[str, Any] | None = None,
        model_spec: Mapping[str, Any] | None = None,
        evaluation_plan: Mapping[str, Any] | None = None,
        metadata: Mapping[str, Any] | None = None,
        expected_revision: int | None = None,
    ) -> dict[str, Any]:
        """Publish a complete immutable snapshot (not a partial field merge).

        Callers must supply the intended entire revision; omitted fields are
        empty dictionaries. Completed runs retain links to prior revisions.
        """
        if expected_revision is not None and (type(expected_revision) is not int or expected_revision < 1):
            raise ExperimentStoreError("expected_revision must be a positive integer")
        fields = tuple(_json_object(value, key) for key, value in (
            ("dataset_ref", dataset_ref), ("model_spec", model_spec),
            ("evaluation_plan", evaluation_plan), ("metadata", metadata),
        ))
        now = _utcnow()
        with self._connection(write=True) as db:
            current = db.execute(
                "SELECT current_revision FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
            if current is None:
                raise ExperimentNotFound(f"Unknown experiment: {experiment_id}")
            if expected_revision is not None and current["current_revision"] != expected_revision:
                raise ExperimentStoreError("Experiment revision changed; reload before selecting a dataset")
            revision = current["current_revision"] + 1
            db.execute(
                "INSERT INTO experiment_revisions VALUES (?, ?, ?, ?, ?, ?, ?)",
                (experiment_id, revision, now, *fields),
            )
            db.execute(
                "UPDATE experiments SET current_revision = ?, updated_at = ? WHERE experiment_id = ?",
                (revision, now, experiment_id),
            )
        return self.get_experiment(experiment_id, revision=revision)

    def get_experiment(self, experiment_id: str, *, revision: int | None = None) -> dict[str, Any]:
        with self._connection() as db:
            row = db.execute(
                """
                SELECT e.experiment_id, e.name, e.created_at, r.created_at AS updated_at,
                       r.revision, r.created_at AS revision_created_at,
                       r.dataset_ref_json, r.model_spec_json,
                       r.evaluation_plan_json, r.metadata_json
                FROM experiments e JOIN experiment_revisions r
                  ON e.experiment_id = r.experiment_id
                WHERE e.experiment_id = ? AND r.revision = COALESCE(?, e.current_revision)
                """,
                (experiment_id, revision),
            ).fetchone()
        if row is None:
            raise ExperimentNotFound(f"Unknown experiment/revision: {experiment_id} / {revision}")
        return self._experiment_from_row(row)

    def list_experiments(self, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        limit, offset = _pagination(limit, offset)
        with self._connection() as db:
            rows = db.execute(
                """
                SELECT e.experiment_id, e.name, e.created_at, r.created_at AS updated_at,
                       r.revision, r.created_at AS revision_created_at,
                       r.dataset_ref_json, r.model_spec_json,
                       r.evaluation_plan_json, r.metadata_json
                FROM experiments e JOIN experiment_revisions r
                  ON e.experiment_id = r.experiment_id AND e.current_revision = r.revision
                ORDER BY e.created_at DESC, e.experiment_id DESC LIMIT ? OFFSET ?
                """,
                (limit, offset),
            ).fetchall()
        return [self._experiment_from_row(row) for row in rows]

    def create_run(
        self,
        experiment_id: str,
        *,
        revision: int | None = None,
        seed: int | None = None,
        environment: Mapping[str, Any] | None = None,
        expected_current_revision: int | None = None,
    ) -> dict[str, Any]:
        if expected_current_revision is not None and (type(expected_current_revision) is not int or expected_current_revision < 1):
            raise ExperimentStoreError("expected_current_revision must be a positive integer")
        if seed is not None and (not isinstance(seed, int) or isinstance(seed, bool) or not -(2**63) <= seed < 2**63):
            raise ExperimentStoreError("seed must be a signed 64-bit integer or None")
        environment_json = _json_object(environment, "environment")
        identifier = uuid4().hex
        now = _utcnow()
        with self._connection(write=True) as db:
            experiment = db.execute(
                "SELECT current_revision FROM experiments WHERE experiment_id = ?",
                (experiment_id,),
            ).fetchone()
            if experiment is None:
                raise ExperimentNotFound(f"Unknown experiment: {experiment_id}")
            if expected_current_revision is not None and experiment["current_revision"] != expected_current_revision:
                raise ExperimentStoreError("Experiment revision changed; reload before starting training")
            chosen = experiment["current_revision"] if revision is None else revision
            valid = db.execute(
                "SELECT 1 FROM experiment_revisions WHERE experiment_id = ? AND revision = ?",
                (experiment_id, chosen),
            ).fetchone()
            if valid is None:
                raise ExperimentNotFound(f"Unknown experiment revision: {experiment_id} / {chosen}")
            db.execute(
                "INSERT INTO runs VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (identifier, experiment_id, chosen, "queued", now, None, None, seed, environment_json, None),
            )
        return self.get_run(identifier)

    def get_run(self, run_id: str) -> dict[str, Any]:
        with self._connection() as db:
            row = db.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        if row is None:
            raise ExperimentNotFound(f"Unknown run: {run_id}")
        return self._run_from_row(row)

    def list_runs(self, experiment_id: str, *, limit: int = 100, offset: int = 0) -> list[dict[str, Any]]:
        limit, offset = _pagination(limit, offset)
        with self._connection() as db:
            if db.execute("SELECT 1 FROM experiments WHERE experiment_id = ?", (experiment_id,)).fetchone() is None:
                raise ExperimentNotFound(f"Unknown experiment: {experiment_id}")
            rows = db.execute(
                "SELECT * FROM runs WHERE experiment_id = ? ORDER BY created_at DESC, run_id DESC LIMIT ? OFFSET ?",
                (experiment_id, limit, offset),
            ).fetchall()
        return [self._run_from_row(row) for row in rows]

    def update_run_status(self, run_id: str, status: str, *, error: str | None = None) -> dict[str, Any]:
        """Record a caller-reported state; the store never starts a job itself."""
        if status not in _EXECUTION_STATUSES:
            raise ExperimentStoreError(f"Unsupported run status: {status}")
        if error is not None and (status != "failed" or not isinstance(error, str)):
            raise ExperimentStoreError("error must be text, and is allowed only for failed runs")
        now = _utcnow()
        with self._connection(write=True) as db:
            row = db.execute("SELECT status, started_at FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                raise ExperimentNotFound(f"Unknown run: {run_id}")
            if status not in _TRANSITIONS[row["status"]]:
                raise InvalidRunTransition(f"Cannot change run from {row['status']} to {status}")
            started_at = row["started_at"] or (now if status in {"running", "succeeded", "failed"} else None)
            finished_at = now if status in _TERMINAL else None
            db.execute(
                "UPDATE runs SET status = ?, started_at = ?, finished_at = ?, error = ? WHERE run_id = ?",
                (status, started_at, finished_at, error, run_id),
            )
        return self.get_run(run_id)
