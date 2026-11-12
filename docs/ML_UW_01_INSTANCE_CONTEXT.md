# ML-UW-01 — Project-local workbench instance context

## Purpose and scope

This increment introduces a headless, persistent **one-active-dataset-per-instance**
context and exposes a guarded context API through the public first-party UI factory.
The existing Data Lab selection route is deliberately not modified in this
compatibility build because the deployed `blueprint.py` differs from the
provided source. Until its integration patch is reconciled with the running
checkout, selecting a dataset in the old HTML form does **not** automatically
activate it as the workbench dataset; the explicit activate endpoint does.
It is the foundation for the five-tab workbench in ML-UW-02. This patch **does
not** replace the header, change the XML collection controllers, or introduce
new representation algorithms.

## Ownership

- Existing `ExperimentStore(project_root)` owns `analysis/ml_lab/experiments.sqlite3`.
  Schema v3 adds only `workbench_instances`; existing experiment, revision, run,
  and artifact tables and records are preserved by a nondestructive migration.
- Each workbench instance has an independent unguessable `instance_id`,
  optimistic `generation`, and at most one `experiment_id` + pinned revision.
- The immutable experiment revision already stores source identity, columns,
  SHA-256 and Data Lab provenance. **No duplicate dataset registry or data bytes**
  are introduced by the workbench.
- `get_workbench_instance()` calls existing `verify_selection()` on every active
  read. A missing, edited, external, or provenance-changed source returns
  `status: stale`, a non-authoritative `pinned_dataset` for repair/display, and
  `active_dataset: null`. Callers must never treat `pinned_dataset` as safe data.
- Switching active context increments the generation atomically. A stale
  `expected_generation` is rejected rather than overwriting another tab's
  selection. Archived experiments and runs are not deleted or revised.
- A future multi-instance UI can keep multiple rows in the existing table;
  there is no module-global current-dataset singleton.

## API

Headless functions (exported from `ml_lab.experiments`):

- `create_workbench_instance(store)`
- `get_workbench_instance(store, instance_id)`
- `select_workbench_dataset(store, instance_id, path, expected_generation=..., ...)`
- `activate_workbench_experiment(store, instance_id, experiment_id, expected_generation=..., revision=...)`
- `clear_workbench_dataset(store, instance_id, expected_generation=...)`

First-party mounted UI routes (all project-scoped, no external service):

- `GET /api/workbench/context` creates or reads the current browser's instance.
- `POST /api/workbench/context/activate`: JSON `{experiment_id, expected_generation, revision?}`.
- `POST /api/workbench/context/clear`: JSON `{expected_generation}`.

These are subject to the surrounding host's authentication, if applicable.
A scoped `HttpOnly; SameSite=Lax` cookie retains the opaque context ID between
pages under the mounted blueprint prefix. The context's authoritative data
remain inside project-local SQLite, never inside the cookie.

The existing `POST /data/experiment/selection` is **unchanged** in this
compatibility patch. To bind a selection, call
`POST /api/workbench/context/activate` after an experiment revision has been
created, passing the current `expected_generation`. UI form integration is
intentionally deferred until the live `blueprint.py` is available.

## Migration caveats

- v3 is a forward schema migration. Code built against the prior v2 schema
  cannot simply open a migrated store using an older ML_Lab version; retain
  a backup and coordinate installation upgrades before deploying.
- Selecting a new experiment revision and binding its pointer are separate
  write transactions. A rare simultaneous context switch may leave an
  *unbound but valid* experiment revision, rather than lose existing results.
- The full five-tab navigation and shared dataset controls will be wired in
  ML-UW-02. The XML single-file/collection UX repair remains ML-UW-03.
- The scoped context cookie is not an authorization mechanism. A host exposing
  ML_Lab beyond trusted local users must enforce its own authentication and
  CSRF protection for mutating routes.

## Acceptance

1. Create/select dataset and revisit after reopening the store; identity is
   preserved and source hashes are reverified.
2. Switch or clear the instance's active dataset without deleting historical
   experiments and runs.
3. Reject stale generations and cross-project instance IDs.
4. Upgrade existing v2 SQLite data without dropping old tables/records.
5. Exercise mounted JSON context endpoints without breaking legacy URLs.
6. Use `ml_lab.ui.create_ui_blueprint` (public factory); callers importing
   `ml_lab.ui.blueprint` directly bypass the optional workbench routes until
   the later blueprint-integration patch.
