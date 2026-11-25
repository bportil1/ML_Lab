# SU-04 — ML_Lab shared graph viewer integration

**Status:** Development implementation, pending real Flask/PAH-hosted browser acceptance.
**Source:** `ML_Lab-main(7).zip`; consumes PAH_Shared's **SU-02** `pah-shared.graph-view@1` library without copying its visualization engine.

## User workflow

1. Open an existing ML_Lab instance and go to **Data → Structured data**.
2. Select and fingerprint-verify an **existing project-relative JSON artifact** through the existing `StructuredObject`/source-adapter selector. The five top-level ML_Lab sections, sidebar dataset readiness, and active-tab subpanels remain unchanged.
3. Select **Explore source graph**. No graph renderer loads, and no graph API request is made, until that button is activated.
4. The owner route `/workbench/structured/graph` re-reads the selected project-local artifact and requires its original SHA-256 fingerprint, then emits an exhaustive `pah-shared.graph-view@1` **source_structure** view. The browser imports the actual packaged PAH_Shared `interactive_graph_view.mjs` and mounts it.
5. Use Fit, Zoom, Group overview, Original view and linked node/edge inspector. Close discards the detached preview. The source artifact is not changed, and no analysis or training is run by navigation or display.

### Responsibility split

- **ML_Lab**: validates the source, preserves IDs and original source-reference metadata, decides whether source edges are explicitly eligible for source tracing, supplies presentation groups, verifies source revisions, and provides read-only source inspection.
- **PAH_Shared**: owns actual rendering, camera, layouts, pan/zoom, overview/drill-down, keyboard selection and selection callbacks. Its JS/CSS are served from the **installed local package**, not duplicated or loaded from a CDN.
- **Scientific APIs**: remain separate. This preview does **not** estimate variance, entropy, MI, CMI, statistical associations, probabilistic CPTs or the utility of pruning. These are planned in SU-05–SU-10.

Tree and forest nodes use bounded **display-depth groups** (`depth-00`–`depth-39`; the last group includes deeper nodes). DAG nodes are visually grouped through their validated topological order; this does **not** mean a node has a unique parent. General and cyclic graphs use a circular display. Only verified rooted-hierarchy edges are source-traceable; no DAG, cyclic or undirected link is automatically treated as a causal, statistical or probabilistic relationship. All original IDs remain intact, including multigraph edges. Inspector data is owned by ML_Lab, not added as scientific descriptors to the shared view.

### Important limits

- Existing ML_Lab UI input limit: 2 MiB source JSON, 10,000 aligned observation rows.
- Shared viewer input: at most 5,000 nodes and 15,000 edges. Oversized inputs fail with an explicit message; no silent sampled representation is created. PAH_Shared itself automatically uses visibly labeled overviews for render-heavy but valid graphs.
- Text identity lengths are checked before conversion. Unsupported identities are rejected without truncating or changing source IDs.
- The linked records panel shows only the first **80** records; renderer selection can inspect the full verified object within PAH_Shared's resource limits. This bound is UI-only.
- The projection uses empty metric overlays. Later structure-aware scientific overlays must explicitly identify measured vs unavailable results and use source-attested denominators.
- Selection is stored by existing HttpOnly workspace-scoped cookie containing relative path, adapter and digest. New graph requests must pass source SHA validation. No project databases are shared.
- PAH_Shared **0.3.x** is still allowed for existing ML_Lab UI; without the SU-02 interactive files, the optional preview is unavailable, and existing tools are unaffected. Shared preview requires **PAH_Shared 0.4.x with SU-02 applied**. `static_asset()` does not yet permit the new files; ML_Lab's existing asset endpoint serves only their two exact, package-owned names alongside the existing matrix assets.

### Verification and acceptance

- `pytest -q tests/test_su04_shared_graph_integration.py`
- `node --check src/ml_lab/ui/static/ml_lab_structured_graph.js`
- SU-04 Python projection is exercised against SU-02's **actual** `validateInteractiveGraphView()`, `interactiveGraphRows()`, and `interactiveGraphLayout()` for trees, DAGs and cyclic graphs when a local SU-02 source is present. Tests skip that cross-check when unavailable (no network access or downloaded runtime).
- Existing ML-UW-27–37, sidebar and XML workflow regression selections must remain green.
- **Still required:** installed Flask route and assets, opt-in click/close, pan/zoom, original metadata, large-graph overview, source change, per-workspace cookie isolation and real PAH host browser testing. Absence of Flask or browser test infrastructure is a skip, **not** a pass.

### Next

SU-05 binds original nodes/subtrees to aligned statistical observations, without inventing samples from graph topology. SU-06–07 will attach actually computed branch variance and MI/CMI to the shared viewer using explicit overlay status/units/support. SU-08–10 add controlled pruning/flattening studies and interactive comparison. SU-11 closes acceptance.
