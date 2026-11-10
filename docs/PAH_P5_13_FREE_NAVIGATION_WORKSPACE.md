# P5-13 — ML_Lab optional scientific tool workspace

## Ownership and scope

ML_Lab still owns model families, training, result verification, datasets, XML selection and all experiment metadata. The new navigation shell is a thin UI adapter: no new executor, independent model storage, cross-project index, or numerical algorithm. PAH_Shared continues to own the generic editable matrix and table components already adopted in P5-04.

Use `/workspace` (or the **Workspace** header link) to open Data, Models, Representation, EBM, Optimization, Generative, XML, and Results independently. The left tool palette is present on existing pages. Advanced JSON task routes and the old Overview links remain available. The model workspace no longer presents an ordered, compulsory four-step ribbon; a scientific operation still requires the inputs it has always needed.

Routes remain in the mountable Flask Blueprint. These are *navigation surfaces*, not new scientific implementations:

| Tool | Existing authoritative surface |
|---|---|
| Data | `/data` and its transform/profile/source tools |
| Models | `/experiments/models` |
| Representation | `/experiments/models?family=representation` |
| EBM | `/experiments/models?family=rbm` |
| Optimization | the saved-model optimization section in `/experiments/models` |
| Generative | `/experiments/synthetic` (including GAN/RBM synthetic methods when appropriately configured) |
| XML | `/data/xml/collection` and existing C1–C7 routes |
| Results | new read-only `/workspace/results`, linking to existing verified run comparison and run details |

## Project-scoped, browser-persistent navigation selection

Only the **experiment ID**, **run ID**, and **last tool ID** are stored in browser `localStorage`, keyed by a SHA-256-derived, opaque scope from the resolved project root and blueprint mount name. Raw project-root paths are **not** placed in the storage key or HTML scope attribute. This state is navigation preference, not scientific provenance, a source of truth, or an authorization token. It is per browser profile (not shared across devices or server users). Running in private mode without storage still works; a selection simply does not persist. It does not silently change model revisions, data selection, or run state.

`GET /workspace/context?experiment_id=...&run_id=...` validates both IDs against the **currently bound ML_Lab ExperimentStore**. A run belonging to another experiment is refused; an experiment belonging to another project cannot be resolved. The response includes only display metadata and a bounded list of run IDs/statuses, with `Cache-Control: no-store`. Deleting browser preferences does not delete an experiment or run. A project change produces a different storage namespace. The backend still verifies revisions, source fingerprints, and artifact integrity when scientific operations are requested.

The browser can remember the current experiment when following direct tool links, but direct links also work without JavaScript. A missing selection never blocks the palette or the Results tool; Results offers an explicit experiment picker instead of requiring model configuration.

## Preservation and limitations

- No changes to `ExperimentStore` schema, existing run/provenance files, model execution APIs, or scientific algorithms.
- No auto-training or auto-synthetic generation; user actions remain explicit.
- Old JSON workbenches, old data and XML endpoints, and standalone/PAH-mounted Blueprint paths remain available.
- Existing model, optimization, and synthetic forms still enforce their original scientific preconditions. Free navigation means *opening* a tool does not require completing another tool first, not that a model can train with no data.
- P5-14/15 will address hierarchical XML affordances and typed variables. They are deliberately not combined with this navigation patch.
- Browser-local context is not a durable server-side multi-user preference store. Client-side UI tests require optional Flask; headless tests verify mapping and isolation helpers.

## Local verification

`PYTHONPATH=src pytest -q tests/test_workspace_catalog.py tests/test_workspace_flask.py`

`node --check src/ml_lab/ui/static/ml_lab_workspace.js`
