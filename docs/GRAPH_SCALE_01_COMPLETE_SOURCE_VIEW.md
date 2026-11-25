# GRAPH-SCALE-01 — Complete source graph and progressive inspection

**Repos:** `ML_Lab` (owner of source artifacts, project scoping, inspection),
`PAH_Shared` (single renderer and view contracts). Neither patch changes PAH
host source, embedded `modules/`, pyPIQUE scientific semantics, SQL, nor jobs.

## Removed graph/UI admission limits

- Removed the 5,000-node / 15,000-edge rejection from the ML_Lab source
  presentation projector and PAH_Shared interactive graph validator.
- Removed the 300-display-group ceiling and viewer ID/label character ceilings;
  original source identity remains exact, and label abbreviations are solely a
  legibility affordance with full text in titles/source inspector.
- XML hierarchy presentation no longer rejects an otherwise valid hierarchy
  solely because it contains more than 5,000 entries or exceeds 64 levels.
- Structured Uncertainty file selection no longer rejects a trusted project
  source at 2 MiB; CSV/TSV imports no longer reject a file at 256 columns or
  10,000 record rows. The generic JSON adapter no longer applies an arbitrary
  default node total. Explicit adapter `max_nodes` still works when requested.
- The graph record navigator now supports text search and all-page navigation,
  instead of showing just 80 original nodes/edges.

## How large graphs are displayed

The backend produces **exhaustive** `pah-shared.graph-view@1` data. The shared
renderer validates all source IDs and edge endpoints. For large graphs, it
presents a **labelled collapsed overview** with navigable group/edge pages;
selecting an original ID opens its corresponding page. This is progressive
*rendering*, not statistical subsampling, source truncation, or a transformation.
Original source topology and observer/source revision remain immutable.

Full original records are available from the owner inspector and record
navigator. Group summaries are not original graph nodes and cannot be used as
statistical observations. The initial SVG frame is intentionally kept to a
manageable number of elements for browser responsiveness.

## Memory, safety, and independent scientific limits

The input has no fixed file-byte limit by default. **Large project files may
consume substantial RAM and response time** because source verification,
object normalization and an exhaustive JSON graph projection still happen in
process. Administrators can set `ML_LAB_STRUCTURED_MAX_SOURCE_BYTES` to a
positive integer to enforce a local resource budget (`0` means no byte cap).
Symlink rejection, project-root enforcement, fingerprint checks, XML parser
safety rules, exact source IDs and valid graph structure are unchanged.

**Not silently removed:** separate analysis algorithms have explicit limits on
row×variable work, estimator state spaces, multivariate inference and optimizer
work budgets. For example, the observed-information-network algorithm retains
its 128-variable/10,000-row/10-million pair-row safeguards and the joint MI
estimator has an explicit dimensionality/state-space feasibility policy. Such
limits require batched or out-of-core computation, not merely deleting checks.
Other bounded UI features (such as project-file discovery and older static
hierarchy DOM rendering) likewise require dedicated paginated implementations
to scale without freezing. No current patch claims those are unbounded.

## Acceptance checks

- PAH_Shared `node --test tests/test_interactive_graph_view.mjs tests/test_hierarchy_adapter.mjs`
- ML_Lab `PYTHONPATH=src python -m pytest -q tests/test_large_graph_display_contract.py tests/test_su04_shared_graph_integration.py tests/test_su11_acceptance.py tests/test_xml_shared_hierarchy_p517.py`
- In a PAH-hosted browser, load a graph beyond the former 5k/15k threshold,
  browse group pages and records, select original source node/edge IDs, and
  inspect full metadata. No real browser test was run in this environment.
