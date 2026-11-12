# ML-UW-02 — Research-workbench navigation consolidation

## Scope

The legacy horizontal `Overview | Workspace | Data | Experiments | Compare |
Transform | Provenance` navigation is replaced with a primary **vertical** workflow menu:

1. Data
2. Feature Profiling
3. Representation
4. Comparison
5. General ML Tools

The sidebar contains relevant secondary links and preserves existing scientific-tool
entrypoints (`data-workspace-tool`) inside their corresponding sections. The
existing top header retains ML_Lab identity, version, and one **read-only**
instance dataset-status display. The former `any order` phrase is not used.

No previous route or scientific capability is deleted. Existing Data, XML,
Transform, Provenance, model experiment, synthetic-data, and task URLs remain
available; their behavior is unchanged. Four new read-only section landings
collect links to existing tools while clearly distinguishing **Available**
from **Planned** capabilities. No feature-profiling rewrite, new sklearn method,
or autoencoder implementation is implied by their presence in the navigation.

This patch deliberately does **not** modify `ui/blueprint.py`, which has drifted
in the deployed installation. The public `ml_lab.ui.create_ui_blueprint` factory
registers the four section routes and context processor, independently of the
legacy blueprint module. Applications mounting the legacy blueprint factory
*directly* still render the new sidebar, but use fallback links to existing
routes; section landing pages require the public factory.

## Dataset context limitations

The header's local same-origin GET requests `/api/workbench/context` from
ML-UW-01 and displays the current state. The status is not evidence of data
binding unless the API returns `status=ready` with a verified active dataset.
This increment **does not** bind the old `/data/experiment/selection` form to
the instance. Until that integration, select or activate a compatible experiment
through the ML-UW-01 explicit workbench API. Historical experiment references
are unaffected. A stale dataset reports a warning instead of a usable context.

## Regression checks

- Standalone root and all four new landing pages render without requiring
  PyTorch or new dependencies.
- Prefix-mounted routes and static resources work under PAH-hosted blueprints.
- Legacy links for Data, XML collection, Transform, Provenance and experiments
  still resolve to the existing controllers.
- One sidebar exists; the horizontal secondary navigation is gone.
- The new navigation does not require changes in `blueprint.py`.
- Read-only context status never creates/updates experiment selection.

## Follow-on

ML-UW-03 will reconcile the XML single-file/collection workflow, including
actual collection extraction and data-context activation. Profiles and
representation/comparison tools will be wired to that selected context in
subsequent increments. Future multi-instance management remains deferred.

## Host-level UI boundary

The uploaded v0.32.0 baseline contains `Tools · any order` in its own
`base.html`. This replacement removes that phrase from ML_Lab and keeps only
`Tools`. If a second phrase remains in the PAH host chrome, that separate host
component must be addressed in PAH rather than hidden by module CSS.

## Reconciliation baseline

Rebuilt against the user-supplied `ML_Lab-main(4).zip`, which already includes
ML-UW-00 and ML-UW-01 core compatibility. Unlike the earlier ML-UW-02 patch,
this version preserves P5-13 `data-workspace-*` hooks and the existing
`ml_lab_workspace.js` loader. It does not alter `ui/blueprint.py` or the
project-local SQLite migration. Do not apply either prior ML-UW-02 patch first.
