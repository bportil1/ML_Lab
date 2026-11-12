# ML-UW-14 — Unified Workbench Final Acceptance (internal)

**Baseline:** `ML_Lab-main(4).zip` includes the reconciled ML-UW-01 changes; apply reconciled ML-UW-02 followed by ML-UW-03 through ML-UW-13. This acceptance increment applies **after ML-UW-13**.

**Scope:** Validate the five-tab, source-verified, project-scoped ML_Lab workbench. No scientific engines, training algorithms, scheduling, dataset formats, persistence schema, or public APIs are changed by ML-UW-14. No production behavior is altered to satisfy obsolete tests.

## Acceptance disposition

**Code-level acceptance: PASS on the locally reconstructed baseline.**

**Flask/PAH/browser acceptance: NOT YET VERIFIED in this environment.** Flask is an optional dependency and was unavailable here; browser route tests were skipped. A full project-suite invocation also exceeded the available execution window, so this report does **not** claim complete suite coverage or production deployment acceptance.

### Verified local tests

Command run from the repository root:

```bash
PYTHONPATH=src python -m pytest -q \
  tests/test_ml_uw_*.py \
  tests/test_experiment_model_form.py \
  tests/test_experiment_store.py \
  tests/test_experiment_artifacts.py \
  tests/test_xml_typed_workspace_p515.py \
  tests/test_xml_typed_materialize_p515.py
```

On the reconstructed ML-UW-13 baseline plus this patch: **121 passed, 15 skipped**, with one warning. Skips include tests gated on optional Flask. The additional ML-UW-14 tests alone, together with the corrected model-form tests, yielded **21 passed, 1 skipped**. Counts overlap and must not be added together.

### Acceptance matrix

| Area | Tested contract | Local result |
| --- | --- | --- |
| Unified navigation | Five sections remain in Data → Feature Profiling → Representation → Comparison → General ML Tools order; templates compile | PASS (headless) |
| Dataset handoff | One verified project-local source is shared without implicit model execution or source modification | PASS (headless) |
| Feature profiling | Display-role metadata and cached baseline/preview agree with a full report | PASS (headless) |
| Representation | Registered inverse/out-of-sample capability flags exclude unsupported methods | PASS (headless) |
| Comparison | PCA and SVD consume matching source checksum, columns, and scaling | PASS (headless) |
| Autoencoder configurations | Saving creates an immutable experiment revision, not a training run; historical revision preserved | PASS (headless) |
| Project storage | SQLite schema version remains 3; unrelated project store stays isolated | PASS (headless) |
| XML extraction | Existing typed XML and materialization contracts in targeted regressions | PASS (headless) |
| Experiment artifacts | Existing store/artifact tests in targeted regressions | PASS (headless) |
| PAH-mounted pages | Five live pages, generated URLs, cookie/workbench identity, and hub anchors | NOT EXECUTED (Flask unavailable) |
| Full end-to-end app | Complete suite, actual browser, larger XML collections, benchmarks | NOT VERIFIED |

### Obsolete model-template test corrected

The legacy `test_template_renders_each_model_family_from_registry_without_flask` asserted the literal **“3 · training”** label. This belonged to an earlier numbered workflow and conflicts with the deliberately independent navigation and saved-configuration gate. ML-UW-14 updates the test to require the independent-tools description and to assert that **“Train saved configuration” is absent before any configuration has been saved**. This is a contract correction, not a cosmetic alteration or a suppression of a broken product behavior. No model template or trainer is changed.

## Remaining deployment acceptance gates

Run the same targeted test command in an **already provisioned, offline** environment where Flask, the ML_Lab UI extra, and PAH integration dependencies are present. Use `-rs` to review skipped tests. **Do not treat skipped Flask cases as passing tests.** Then manually verify:

1. The standalone UI and PAH-mounted UI display the five sections and correct mount-prefixed links, including General ML Tools subsection anchors.
2. A selected dataset appears consistently in Feature Profiling, Representation, Comparison, and General ML Tools; another browser tab's stale form is rejected without losing user data.
3. CSV and XML single-file/collection results have explicit handoff into Data, and collection return navigation preserves filters and selected directory.
4. Saving an MLP/Transformer autoencoder configuration does not start training; an explicit run uses the revision-pinned configuration and stores a project-local verified artifact.
5. The legacy Models, EBM, Optimization, Generative, experiment history and artifact UIs remain accessible. Non-invertible representation methods never expose a working reconstruction operation.
6. Verify rollback from a disposable clone; do not remove datasets, databases, experiment revisions, or artifacts to roll back code changes.
7. Measure cold and warmed feature-profiling performance and large XML collections on representative data before claiming throughput improvements.

### Known deferred items (not release blockers for source-level consolidation)

- Synchronous XML collection review is bounded; larger collections still require deliberate scalability work.
- Full independently held-out representation benchmarking is not implemented; in-sample metrics must retain their current labels.
- Full application benchmarks, long-running-job management, and project-wide Flask/PAH runtime tests remain unverified.
- Neural and sklearn method-specific convergence/runtime warnings depend on data and optional environment support.

## Rollback

This patch modifies **tests and internal documentation only**. Reverting ML-UW-14 does not alter project SQLite data, source files, models, UI routes, or artifacts. Revert the source changes in reverse order if rolling back earlier ML-UW increments. Do not delete project-local data during rollback.
