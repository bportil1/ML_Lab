# ML-UW-10 — Unified workbench integration and saved-revision handoff

Internal implementation and regression notes; no external project material is required.

## Purpose and ownership

The unified navigation keeps Data → Feature Profiling → Representation → Comparison → General ML Tools in one module. All five sections use the same `ExperimentStore` and the **browser workbench's** pinned experiment revision; they do not migrate scientific engines or duplicate project data. Historical runs, artifacts, and model revisions continue to belong to their original experiments.

The General ML Tools catalog lists existing project experiments, but opening an experiment's **model editor** does **not** switch the active dataset. To make that boundary explicit, each saved experiment in the Models section also offers **Use in workbench**. That link previews the verified revision. The POST confirmation is the only action that binds it to the workbench instance. No experiment revision is created by this handoff; opening the preview neither starts a run nor changes the binding.

## HTTP behavior

- `GET /workbench/experiment/activate?experiment_id=…&revision=…`: project-local experiment lookup, source/provenance verification, read-only preview and current generation.
- `POST /workbench/experiment/activate`: explicit `experiment_id`, `revision`, and `expected_generation`; delegates to `activate_workbench_experiment` with optimistic generation checking, verifies selection again, and returns a confirmation. An already-active matching revision is idempotent and does not advance the generation.
- Any failed source, lineage, identity, or project check produces an error instead of silently rebinding the workbench. A concurrent tab change returns HTTP 409.
- Data-selection and archived runs remain available. `experiment_model` still only edits its own chosen model configuration.
- Browser status uses `url_for('…workbench_context_api')` so the context endpoint respects the mounted PAH prefix, rather than constructing paths from string fragments. The legacy direct blueprint has no context API and does not claim otherwise.

## Regression contracts

`tests/test_ml_uw_10_integration.py` covers:

1. Shared source identity between Data, feature-role metadata, PCA/SVD comparison, and autoencoder model configuration; no retraining from navigation.
2. Switching datasets, reactivating a saved revision, preserving historical model specs and restoring roles only when matching dataset fingerprints.
3. Rejection of stale source bytes, cross-project experiment IDs and stale generations.
4. Template parsing, explicit form submission, contextual links and status URL wiring.
5. Optional Flask-mounted browser path tests (requires `ml-lab[ui]`): preview-before-activation, idempotent confirmation, all five sections, feature profiling, PCA, cross-method comparison and independent browser instances.

The headless suite is available in base/offline installations. The mounted-UI check is skipped if Flask is absent; in that case no browser success is claimed. This patch does not add dependencies or migrate SQLite (`PRAGMA user_version` remains 3), and does not introduce a new training engine or background workers.

## Reversion

This increment modifies only the workbench routes/navigation, their templates, tests and this internal document. Removing it does not change experiment records, run artifacts, dataset files or source provenance. A revision activated through this page uses the same preexisting `workbench_instances` pointer mechanism as the earlier context API.
