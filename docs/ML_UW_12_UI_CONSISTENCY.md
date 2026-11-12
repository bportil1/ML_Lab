# ML-UW-12 — UI consistency, error recovery, and stale form notices (internal)

## Scope

ML-UW-12 is a stabilization patch on top of ML-UW-11. It does not change
scientific calculations, extraction behavior, model engines, experiment
revisions, project storage, APIs, or the existing 5-section navigation.

- A common Jinja error component displays an accessible error plus an explicit
  navigation/reload path for the new workbench views. Reload is a read-only GET;
  it discards unsubmitted form edits and does not retry any action.
- Representation errors remain visible even when the selected dataset becomes
  unavailable while the form is open.
- The dataset-selection endpoint returns 409 for `WorkbenchConflict` based on
  exception type, not the wording of an error message. Ordinary validation
  problems remain 400. No error causes automatic dataset activation.
- The header's read-only context fetch refreshes on initial load, `pageshow`,
  and visible-tab return. There is no timer and no background polling.
- Forms with an `expected_generation` show a client-side warning if their
  pinned workbench generation no longer matches. Forms that are meant to select
  a new source do **not** compare that candidate SHA to the active dataset SHA.
  A warning does not change hidden inputs, block the form, discard changes,
  train a model, or bypass server-side optimistic concurrency checks.
- Keyboard users can skip directly to the main area. Form controls and errors
  have visible focus and accessible status styles.

## Acceptance / current limitations

- Jinja templates compile and representative workbench pages render using
  the local Jinja runtime; the new context JS passes Node syntax and mock-DOM
  behavior tests.
- ML-UW and targeted XML/UI regressions pass in the isolated environment.
- Optional Flask-dependent route/browser tests are skipped when Flask is not
  installed. Do not treat mocked browser tests as a substitute for testing a
  real PAH-mounted Flask interface.
- No automatic form mutation means another tab can still change the context
  after the check: the server's `expected_generation` and source fingerprint
  checks remain authoritative.
- Existing Data Lab and XML legacy layouts keep their established error
  rendering; the shared component is used on the new unified views only.

## Rollback

The patch is source-only. Reverting it does not require any database migration
or artifact cleanup. Keep previous patch order through ML-UW-11; after reverting
ML-UW-12, the existing workbench dataset selections and experiments remain.
