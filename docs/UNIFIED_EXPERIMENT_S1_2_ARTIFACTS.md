# Unified Experiment — S1.2 Artifact Registry and Provenance

**Implementation:** `ml_lab.experiments.ExperimentStore(...).artifacts` (opt-in,
headless Python API). This sprint does not change the UI, CLI, model dispatch,
training engines, Data Lab transform records, or the PAH runtime integration.

## Storage, compatibility, and migration

The local project database remains
`<project_root>/analysis/ml_lab/experiments.sqlite3`. On opening an existing
**S1.1 schema v1 database**, ML_Lab performs an atomic additive migration to
**schema v2**. Existing experiment revisions, run IDs, run states, and input
snapshots are retained. This is a forward migration: older S1.1 code refuses
the newer schema; back up the SQLite file before upgrading valuable projects. New databases initialize with the same experiment/run
schema plus these additional tables:

- `artifacts`: file reference identity, producer/consumer run, `input`/`output`
  role, kind, logical name, canonical project-relative path, optional media
  type, file size, SHA-256, registration time, and JSON metadata.
- `artifact_sources`: ordered references to **already registered artifacts**
  within this same ML_Lab project. Direct sources establish an explicit
  derivation edge; no source link is guessed from similar filenames.

Records describe files at registration time, not an immutable file store or
permanent proof that files will continue to exist unchanged. Model binaries,
large arrays, datasets, and visualizations remain in their normal file formats
outside SQLite. The registry does not move, create, overwrite, or delete them.

## Example — source and derived dataset

```python
from ml_lab.experiments import ExperimentStore

store = ExperimentStore("/path/to/project")
experiment = store.create_experiment("Example model", model_spec={"family": "rbm"})
experiment_id = experiment["experiment_id"]
source_run = store.create_run(experiment_id)

# These are existing files under the selected project directory.
source = store.artifacts.register(
    source_run["run_id"],
    "datasets/original.csv",
    role="input", kind="dataset", media_type="text/csv",
)

training_run = store.create_run(experiment_id, seed=42)
checkpoint = store.artifacts.register(
    training_run["run_id"],
    "analysis/models/checkpoint.pt",
    kind="checkpoint", role="output", media_type="application/octet-stream",
    metadata={"epoch": 100},
    source_artifact_ids=[source["artifact_id"]],
)

assert store.artifacts.verify(checkpoint["artifact_id"])["status"] == "valid"
```

This is an API illustration, **not** a claim that existing training tasks
register their outputs automatically. S1.3 introduces task integration.

## API and public record

- `store.artifacts.register(run_id, path, *, name=None, role="output",
  kind="other", media_type=None, metadata=None, source_artifact_ids=())`
- `store.artifacts.get(artifact_id)`
- `store.artifacts.list_for_run(run_id, limit=100, offset=0)`
- `store.artifacts.list_for_experiment(experiment_id, limit=100, offset=0)`
- `store.artifacts.verify(artifact_id)`

Artifact records use `ml-lab.artifact@1` and include `artifact_id`, `run_id`,
`experiment_id`, pinned experiment `revision`, `role`, `kind`, `name`, `path`,
`media_type`, `size_bytes`, `sha256`, `created_at`, `metadata`, and
`source_artifact_ids`. Acceptable `role` values are `input` and `output`.
Kinds are lowercase identifiers and may be specialized by scientific engines;
typical examples are `dataset`, `checkpoint`, `model`, `metrics`,
`representation`, and `visualization`. Arbitrary structured metadata is
allowed if JSON-compatible; the registry does not pretend that metadata
validates model-specific file contents.

Files are hashed in streaming chunks, with a best-effort check that the
file did not change during hashing. `verify` recomputes the fingerprint and
reports `valid`, `changed`, `missing`, `unsafe`, or `unreadable`; it **does not**
silently overwrite the recorded SHA-256. Source references are checked
transactionally and may link artifacts belonging to other runs **within the
same project**, but never to another project's registry. Since source links
only point to pre-existing artifacts, public registration cannot create a
cycle. The database stores input/derivation connections, but does not parse or
supplant the existing Data Lab/ XML authoritative provenance sidecars.

## Trust and ownership boundaries

- Only existing regular files beneath the selected project root are accepted;
  `..` traversal, files outside the project, and symlinked path components
  are rejected. Paths are stored relative to the project.
- Data remains owned by the module and project. PAH must consume an eventual
  public API/manifest, not query ML_Lab's private SQLite database.
- Registering a file is a **snapshot of metadata**, not file locking,
  transactional content capture, a copy-on-write storage service, or a claim
  that another process cannot later modify it. Executors should finish writing
  before registration. `verify` explicitly detects later changes or losses.
- Artifact registration does not advance run status, schedule jobs, or produce
  checkpoint/resume behavior. Experiment/run state transitions remain those
  from S1.1.
- The S1.2 API is intentionally independent of Flask, PAH, NumPy, PyTorch,
  and remote services. It does not import pyPIQUE or HSQA_DBN; tangent-manifold
  generation remains owned by HSQA_DBN.

## Validation

Run locally:

```bash
PYTHONPATH=src python -m pytest -q tests/test_experiment_store.py tests/test_experiment_artifacts.py
```

Tests include S1.1 run/config preservation through the v1→v2 migration,
source linkage, hash integrity, missing and altered files, symlink and
cross-project rejection, invalid inputs, concurrent registrations, and bounded
pagination. S1.3 will adapt existing task results into these records; there
is currently no automated inventory of old output folders or registration
of historical files.

## S1.3 integration

The opt-in `execute_recorded_task` wrapper now uses this registry for input
payloads, task responses, and known project-local Data Lab derivative outputs.
See [S1.3 legacy adapter](UNIFIED_EXPERIMENT_S1_3_LEGACY_ADAPTER.md).
Existing `execute_task` calls are not automatically recorded.
