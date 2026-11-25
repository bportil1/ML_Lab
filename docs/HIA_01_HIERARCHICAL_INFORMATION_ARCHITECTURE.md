# HIA-01 — Hierarchical Information Analysis: audit and scientific contracts

**Owner:** `~/cluster/ML_Lab` (standalone repository). **Patch:** HIA-01 of HIA-01–12.
**Type:** additive, headless versioned contracts and audit; **no estimator, Flask route, graph/UI, package installation, SQLite, or source-file mutation**. The operational level map is HIA-02, not HIA-01.

## 1. Why another contract is needed

Existing Structured Uncertainty separates these established artifacts:

| Existing source | Responsibility | What it does **not** establish |
| --- | --- | --- |
| `structured.object.StructuredObject` (`ml-lab.structured-object@1`) | Canonical immutable source nodes, edges, observation cells and verified topology capabilities | One observation per graph node, statistical/causal edges, arbitrary hierarchy levels |
| SU-05 `structured.bindings` (`ml-lab.observation-binding@1`) | Declared observation-column→exact source node mapping, repeated-cell policy, group/weight metadata, source digest | Independent samples, statistical graph structure, source identity of merged columns |
| SU-06 `structured.branch_uncertainty` | Source-scoped observed descriptive distributions and numerics, declared group variance | Information propagation or level estimands |
| SU-07 `structured.branch_information` | Empirical-discrete branch information and incremental target relevance on aligned observations | Parent-conditioned information implied by source edges, additive per-node MI |
| ML-UW-32 `structured.information` | Single-variable/pair MI/CMI over aligned observations with explicit numeric discretization | Hierarchy membership, independence or joint information across entire levels |
| NETWORK-01 `structured.network_information` | Exhaustive bounded observed-variable pair matrix and optional bounded joint summary | Source-node association mapping, full probabilistic model or hierarchy |
| SU-04 / SU-10 ML_Lab UI routes and scripts | Source graph/inspector, selected-branch sandbox, opt-in read-only operations | Selecting an entire verified level and explaining its incremental information |
| PAH_Shared `pah-shared.graph-view@1` | Shared graph rendering, layout, selection, zoom, pan, overview, drill down | Scientific inference and ML_Lab-specific estimands |

HIA adds **one scientific selection identity** to reuse these components. A click on a source node is not evidence of MI with nearby source nodes. Graph distance does not establish stochastic or conditional independence.

## 2. New versioned artifacts

Implemented in `ml_lab.structured.hierarchical_contracts`:

- `ml-lab.hierarchical-information-request@1`: strict, immutable `HierarchicalAnalysisRequest` and `HierarchicalScope` with lossless canonical JSON, schema and exact source/binding digests. Reject unknown fields, duplicate JSON keys, nonfinite constants, unsupported scope/metric pairings, ambiguous IDs, invalid conditioning, and implicit independence assumptions.
- `ml-lab.hierarchical-information-eligibility@1`: pure `inspect_hierarchical_eligibility(source, request, binding)` result. Each metric has a `status` (`deferred` or `unavailable`), `reason`, `estimate: null`, and `estimator_support: null`. This gate computes **zero** statistics and should never be displayed as a measured value.

`source_digest` is the SHA-256 of the complete validated `StructuredObject`, not a user-provided `object_id`; `binding_digest` is the SHA-256 of the complete SU-05 binding report. The result checks both digests. IDs are exact originals and are not matched by label or inferred from structural parentage. No new source or observation ID is generated.

### Structural selection

- `node`: exactly one original source node ID; statistics would describe variables explicitly mapped directly to it.
- `branch`: exactly one original root node ID; **only** verified `tree`/`forest` can use descendant membership from SU-05. DAG, cyclic, undirected and generic directed graphs cannot silently borrow tree subtree semantics.
- `level`: one nonnegative integer depth, no node IDs; requires a deterministic `level_policy`. HIA-01 checks policy/topology compatibility but does **not** produce a level map.
- `network`: no node IDs or depth; all future global statistics must use separately identified aligned variables, not all source nodes treated as independent samples.

For HIA-02: `rooted_depth` on tree/forest means minimum unique-parent root distance (each root at depth 0). `dag_longest_path` on an explicit DAG means longest directed path from any source indegree-zero root, breaking no ties by UI ordering. Roots belong to depth 0. A DAG may have multiple parents and edges skipping display levels; this is **not** a claim of conditional independence. Cyclic/undirected/general directed graphs have no default level model and fail closed pending a future explicitly justified policy. All original node/edge identities remain present in source. HIA-02 must construct and verify the complete deterministic level map before rendering or estimating level-scoped reports.

### Metric intents (no estimates in HIA-01)

| Scale | Metric keys | Later stage |
| --- | --- | --- |
| Atomic source node | `node_structure`, `node_descriptive`, `node_neighbor_associations` | HIA-03–05 |
| Branch | `branch_structure`, `branch_joint_entropy` | HIA-02/06 |
| Level | `level_joint_entropy`, `level_conditional_entropy`, `level_target_increment`, `level_factorized_conditional_entropy` | HIA-07–09 |
| Whole network | `network_structure`, `network_information` | HIA-10 |

Neighborhood basis is **explicitly** `structural` (original links) or `statistical` (an association result with support and estimator provenance); source adjacency alone is not MI. Future HIA-05 must distinguish Pearson/Spearman eligibility (typed numeric values) from discrete MI/CMI eligibility and report both pairwise and common-cohort support where appropriate.

## 3. Conditional information accounting

Let `L_k` be the distinct observed-variable IDs mapped to **original structural nodes at depth k**, under a verified level definition and explicit variable ownership/overlap policy. It is not the set of graph nodes treated as samples. Let `P_k = L_0 ∪ ··· ∪ L_{k-1}` after the overlap policy is applied. Let `Y` be a separately specified target with disjoint role.

- **Unconditional level uncertainty:** `H(L_k)` estimated jointly, not by adding marginal entropies.
- **Residual uncertainty conditional on previous levels:** `H(L_k | P_k)`.
- **Already shared with previous levels:** `I(L_k;P_k) = H(L_k) − H(L_k|P_k)`.
- **Incremental target relevance:** `I(Y;L_k | P_k)`; this quantity does **not** generally equal a sum of nodewise CMI values.
- **Assumption-dependent factorization:** If the user explicitly acknowledges `conditioned_factorization`, approximate `P(L_k | P_k) ≈ ∏_{X∈L_k} P(X|P_k)`. It follows that the modeled entropy becomes `Σ_{X∈L_k} H(X|P_k)`; this does not imply exact empirical equality. The empirical–factorized gap (conditional total correlation) is separately measurable where the common cohort is sufficient.

A *single shared included-row cohort*, fixed missingness policy, and fixed categorical/discretization definitions are required for any numerical comparison or telescoping identity. Different pairwise complete-case populations cannot be added into a level-wide information flow. Overlapping variables across levels must be disclosed and resolved by an explicit policy before additive attribution. Source topology never supplies probability model parents automatically. Finite-sample plug-in bias, sparse strata, high cardinality, binning leakage, sampling dependence, repeat occurrences, observation weights, and multi-source attribution must be visible when the estimators become available.

### Assumption gates already enforced by HIA-01

- A conditional metric must declare `conditioning_scope` of `prior_levels` or `explicit_variables`; no default `Z` is fabricated.
- `target_variable` is explicit and only accepted with `level_target_increment`, never silently chosen from source labels.
- `level_factorized_conditional_entropy` requires `independence_model='conditioned_factorization'`, `conditioning_scope='prior_levels'`, `acknowledge_assumption=True`, and a level scope. This **does not** certify an independence hypothesis; it records a modeling choice for HIA-09.
- For unknown binding variables, source-digest mismatch, and merged-node attributions, the gate rejects or reports unavailability. An unavailable estimate is `null`, not `0`.

## 4. Scientific/reporting responsibilities for later patches

HIA-02 must yield a source-pinned, reproducible level map and scope membership, including roots, cross-parent DAG cases, level counts, and O(V+E) or better traversal for deep/large graphs. It must not impose a new 5,000-node visual rejection. Where a source node maps to multiple observed variables, or a variable maps ambiguously to multiple nodes, HIA must preserve that ambiguity and refuse unsupported per-node allocation.

HIA-03 must reuse the **actual** PAH_Shared viewer. Source nodes and edges, statistical association overlays, and explicit PGM edges are separate layers; statistics must be owned and cached by ML_Lab, selected by exact identities. Preserve existing fit/zoom/pan, selection, source trace, graph export and progressive view behavior; make node/branch/level selection discoverable rather than requiring manual data-entry forms.

HIA-04–06 must publish atomic neighbor and branch reports with estimator metadata, sample support, missingness, grouping, repetition policy, and provenance. HIA-07–09 must implement joint and conditional information without presuming additive node contributions or independent parents. HIA-10–11 must offer whole-network comparisons/overlays without silently filtering the underlying matrix. HIA-12 validates all of the above with synthetic XOR, exact independence, redundant columns, DAG multi-parent cases, large hierarchies, source-only XML, collections of aligned records, and local PAH browser acceptance.

For high-dimensional joint estimates, an explicit estimator budget or an unavailable/support status is preferable to truncating nodes or fabricating an exact joint distribution. GRAPH-SCALE-01 removes fixed *display* rejections; it is not a promise of unbounded exact CMI computation.

## 5. Current use and acceptance

```python
from ml_lab.structured import (
    HierarchicalScope, HierarchicalAnalysisRequest, source_sha256,
    inspect_hierarchical_eligibility,
)

# source is an existing validated StructuredObject.
request = HierarchicalAnalysisRequest(
    source_digest=source_sha256(source),
    scope=HierarchicalScope('node', ('actual-source-node-id',)),
    metrics=('node_structure', 'node_descriptive'),
)
report = inspect_hierarchical_eligibility(source, request)
# report metrics are deferred/unavailable by design, not computed estimates.
```

`PYTHONPATH=src python -m pytest -q tests/test_hia01_hierarchical_contracts.py`

**Expected HIA-01 acceptance:** schema roundtrip, strict invalid-request rejection, source/binding pinning, source-only reporting, no invented estimates, unique node IDs, tree/forest subtree membership only when validated, deterministic level-policy declaration, explicit conditional-independence consent, and unmodified source data. No UI change is asserted until HIA-03; no statistical analysis until later stages. **All work belongs in `~/cluster/ML_Lab`; no PAH embedded `modules/` edits.**
