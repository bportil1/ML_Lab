# ML-LAB-MI-01 — PAH independent UI launch compatibility

## Ownership boundary

ML_Lab exposes optional `MLLabRuntimeAdapter.process_launch_spec(context=...)` for
PAH MI-06. The returned JSON-like specification contains only a trusted local
Python command, environment, and UI port request. PAH allocates and owns the
process, runtime directory, port, instance ID, and shutdown; ML_Lab owns its
Flask UI, dataset selection, SQLite experiment store, output artifacts, and jobs.
There is no import of PAH Core in the independent process entry point.

## Paths and identity

- Child executable: `sys.executable -m ml_lab.process_ui`.
- PAH supplies `PAH_MODULE_INSTANCE_ID`, `PAH_MODULE_RUNTIME_DIR`, and
  `PAH_MODULE_PORT`; the entry point requires all three and binds only to
  `127.0.0.1` with the development reloader disabled.
- With a selected host workspace, ML_Lab receives the workspace's project and
  results paths through `ML_LAB_PROJECT_ROOT` and `ML_LAB_RESULTS_ROOT`.
- With no selected project, ML_Lab uses its private PAH runtime directory as
  the project root; no generated analysis output is directed to PAH's cwd.
- Profile and derived-dataset default outputs use
  `<results>/instances/<sha256(instance_id)[:24]>/data/` to prevent sibling
  processes from choosing the same implicit filenames.
- The experiment SQLite database remains project-owned at
  `<project>/analysis/ml_lab/experiments.sqlite3`. It intentionally shares a
  project across instances, using transactions and distinct workbench records.
- The workbench cookie includes the PAH instance identity. Without this,
  two `127.0.0.1:<different port>` UIs for the same project would send the
  same browser cookie and share an active-dataset selection.
- Existing in-process `launch()` and standalone CLI `ml-lab ui` are unchanged.

## Limitations and follow-up checks

- ML Lab must be installed with its optional `ui` dependencies for real
  browser integration; this patch does not install third-party packages.
- Ports are allocated before the process binds, so an external port takeover
  race remains theoretically possible until PAH has an inherited-socket
  readiness contract.
- Explicit user-selected output paths, XML export defaults, and the shared
  project experiment database are *not* exclusively per-process. Working on
  the same project in two windows must use normal storage conflict protections.
- Background telemetry from child processes requires the public publisher
  hook supported by PAH MI-07; launching alone does not register jobs.

## Local checks

`PYTHONPATH=src pytest -q tests/test_ml_lab_mi_01.py tests/test_pah_integration.py`

For end-to-end validation inside PAH: launch two ML_Lab copies under the same
project, verify two local URLs/ports, separate workbench cookies, distinct
output roots, and that stopping one process leaves the other usable. Repeat
with two different project roots and verify no cross-project writes.
