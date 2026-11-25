# HIA-03 — Interactive hierarchy workspace over PAH_Shared

**Status:** Additive UI + verified read-only structural endpoints. HIA-01/02 contracts are required. ML_Lab is the standalone owner; the PAH_Shared renderer is not copied, patched or replaced.

## User-facing workflow

1. Select a local project source from **Data → Structured uncertainty** using the existing source-intake form. Press **Explore source graph**. The original PAH_Shared viewer, metadata inspector, linked-record navigator, zoom/fit/group overview, source-linked sandbox and other tabs are retained.
2. For a verified `tree` or `forest`, the **Hierarchy navigator** offers original-node, rooted-branch, level and entire-network scopes. For a verified `dag`, it offers original-node, longest-path level and network scopes, but *not* unique-parent branch selection. For other graph topology types, the existing graph still opens; the panel explains why level semantics are unavailable.
3. Select an original node on the SVG or record list to inspect its exact parent/child context and HIA-01 metric eligibility. Use **Inspect branch** to select that node and its unique-parent descendants, **Inspect level** for a depth, or **Inspect network** for all verified source nodes. Scope members are paged in batches of 80; all original identities remain in the exported selection report, not just the visible buttons.
4. Click any paged member to recenter the existing PAH_Shared graph on its original node ID. The visible nodes in the chosen scope receive a distinct orange outline; the currently selected node remains blue. The original graph data is not rewritten, filtered or sampled by hierarchy selection.
5. **Fullscreen** expands the existing graph workspace. **New window** opens a separate same-origin page for the same SHA-verified project-local source; it loads the original graph afresh rather than serializing source data or identifiers into a URL. Original selection and camera are not transferred to the new window in this patch.
6. **Export complete graph JSON** contains the full existing source graph, original inspector metadata and HIA overview, without partial-page filtering. **Export selected scope JSON** contains complete, SHA-pinned HIA-02 identity membership and deferred metric eligibility. **Export visible SVG** contains the *currently rendered page* with portable self-contained text and edge styles; it is labeled as a visible-page export, not a complete giant-network export.

## Versioned read-only endpoints

- `GET /workbench/structured/hierarchy`: `ml-lab.hia-workspace-overview@1`. Reports verified level counts, roots, topology policy, source revision, and whether hierarchical scopes are available. Returns an unavailable marker rather than fabricated levels for cyclic, undirected or general graph sources.
- `POST /workbench/structured/hierarchy/inspect`: `ml-lab.hia-workspace-inspection@1`. Requires the same local SHA-verified selection cookie used by SU-04, JSON fields `source_revision`, exact HIA-01 `scope`, nonnegative integer `offset`, and `X-ML-Lab-HIA: explicit-inspection`. Reports the complete HIA-02 selection plus a bounded *rendered member page*.

Each request reopens and validates the project-local source against its file SHA-256 and uses HIA's separate normalized `source_sha256` digest. A stale artifact or mismatched request fails closed. The endpoints do not access unrelated project DBs, create analysis jobs, write artifacts, or establish any statistical relationship from source structure.

## Scientific boundary

The graph displays **source topology**, not learned or probabilistic edges. HIA-03 does not calculate entropy, MI, CMI, Pearson/Spearman, node variance, joint level statistics or any pruning decision. The existing HIA-01 eligibility record may say *deferred* or *unavailable* and carries no fake numerical value. SU-05 bindings still require explicit user action through the existing evidence sandbox. HIA-04/05 add node/neighbor statistics and HIA-06–09 add branch/level and conditional information.

## Implementation/ownership

- `src/ml_lab/ui/structured_hia_view.py`: projection of HIA-02 results into source-verified overview and inspect responses. Scope paging is for HTML controls only; it never caps or reassigns source identities.
- `src/ml_lab/ui/workbench_structured_routes.py`: scoped read-only endpoints, consistent SHA checks, cache fingerprint includes the new controller.
- `src/ml_lab/ui/static/ml_lab_hia_workspace.js`: additive event listener on `mlwb:source-open`, `mlwb:source-reset`, `mlwb:graph-select`, `mlwb:graph-viewport`. The existing PAH_Shared `viewer` is passed by the existing owner controller; the new code never draws a competing SVG engine.
- `src/ml_lab/ui/templates/ml_lab_ui/workbench_structured.html` and `ml_lab_workbench.css`: accessible, high-contrast scope controls and inspector.
- `src/ml_lab/ui/structured_graph_view.py`: remove the legacy depth-39 display-group clamp so the shared viewer can navigate the exact HIA-02 levels; the viewer still uses progressive rendering for large networks.
- `tests/test_su11_acceptance.py`: revise the legacy 40-group assertion to require original deep level identities.
- `tests/test_hia03_interactive_workspace.py`: identity, topology policy, large-graph pages, malformed request and frontend contract checks, plus conditional Flask tests where Flask is installed.

No changes to `Project_Assistant_Host/modules/*`, any standalone sibling repository, source files selected for analysis, or scientific databases.

## Test and manual acceptance

```bash
PYTHONPATH=src python -m pytest -q tests/test_hia01_hierarchical_contracts.py tests/test_hia02_hierarchical_index.py tests/test_hia03_interactive_workspace.py tests/test_su04_shared_graph_integration.py
node --check src/ml_lab/ui/static/ml_lab_hia_workspace.js
node --check src/ml_lab/ui/static/ml_lab_structured_graph.js
```

Live hosted-browser acceptance is still needed for fullscreen/new-window browser behavior, long labels, detached-window cookie scope and large-graph navigation. Keep old SU-10 sandbox features intact and confirm that source reset clears the hierarchy selection.
