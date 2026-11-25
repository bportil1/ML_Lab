# HIA-07 — Whole-level empirical information (explicit observed subset)

## Scope, ownership and compatibility

**Owner:** standalone `~/cluster/ML_Lab` source repository. This is patch 7
of the HIA-01–12 sprint. No changes to PAH's embedded `modules/` directory,
PAH_Shared, pyPIQUE, HSQA_DBN, or any project SQLite data or analysis outputs.
The existing PAH_Shared interactive renderer remains the sole graph engine.
All HIA-01–06 inspectors, SU-01–11 data workflows, and complete original source
node/edge identities remain intact.

**User workflow:** Open **ML_Lab → Data → Structured Uncertainty**, select a
verified source, press **Explore source graph**, choose **Inspect level** in
the HIA-03 hierarchy navigator, and open **Level information**. Original
members can be chosen from any page of the hierarchy's source-ID inventory.
Explicitly associate each chosen original member with a *distinct* observed
column; choose the handling of repeated cells and optional numeric bins;
then press **Compute mapped level information**. Calculation never occurs
on mere graph open, hover, node selection, or level selection. The complete
scientific report can be exported as JSON; no database or source modification
occurs.

## Structural level versus statistical scope

An exact structural depth is computed by HIA-02:

- Verified trees and forests: `rooted_depth` from a source root; each root
  has depth zero, nodes have a unique parent.
- Verified DAGs: `dag_longest_path` from *any* indegree-zero root, with
  multi-parent reachability retained; DAG depths are not learned layers.
- Undirected, cyclic, or general directed graphs have **no implicit level
  semantics** and remain ineligible for level analysis.

`level_node_ids` contains **every original source node at depth `level`**
and is not silently sampled, truncated or altered to match observed columns.
`preceding_level_node_count` describes preceding **structural** scope only;
it is **not** an information estimate or a conditioning set in this patch.

`mappings` declares the subset of level nodes having explicit, one-to-one
SU-05 observation bindings. Each source node ID and each observed column is
unique in this request. `mapped_node_count`, `unmapped_node_count`,
`complete_level_statistical_coverage` (mapping coverage), and
`statistical_scope` distinguish a complete mapping from a partial one.
**Mapping coverage does not guarantee estimation succeeded**; check
`level_information.status` and `support` separately. A source-only tree may
have a perfectly valid level inventory but zero samples and no valid entropy.

## Empirical estimands

Write the *mapped observed-variable vector* for one level as
\(L=(X_1,\ldots,X_k)\). This is a definition over explicit observation
columns, never a conversion of source graph edges to statistical edges.
All quantities below use the **same set of complete observation rows**:

- `joint_entropy_bits`: \(H(L)\), categorical/discretized plug-in entropy.
- `sum_atomic_entropy_bits`: \(\sum_i H(X_i)\), marginal entropies on the
  **same common cohort**.
- `total_correlation_bits`: \(TC(L)=\sum_i H(X_i)-H(L)\).
  It is not a sum of pairwise mutual informations and is not a causal claim.
- `per_mapped_node[].entropy_bits`: \(H(X_i)\).
- `per_mapped_node[].conditional_entropy_given_other_mapped_members_bits`:
  \(H(X_i\mid L\setminus\{X_i\})\).
- `per_mapped_node[].mi_with_other_mapped_members_bits`:
  \(I(X_i;L\setminus\{X_i\})\). These overlapping nodewise MI numbers
  **cannot be summed** to recover total or unique shared information.

For a single mapped variable, total correlation and its MI with the empty
complement are zero. All measures are in bits. Empty or unsupported cohorts
return a **status and null statistics**, not invented zeros.

The report includes: observation-unit type, source and binding fingerprints,
complete source-level inventory, exactly declared mappings, full binding plan,
common sample support, excluded row counts/reasons, discretization policy,
number of distinct joint states and a high-dimensional sparsity warning.
Missingness exclusions are visible; unresolved repeated cells disable the
estimates unless a user explicitly selects an appropriate aggregation policy.
Floating-point values require explicit numeric quantile/equal-width bins.

**Scientific limitations:** empirical plug-in entropy may have substantial
finite-sample and sparse-state bias. Large source graphs do not automatically
create statistically independent samples. Level-to-level conditional entropy
\(H(L_k\mid L_{<k})\), target incremental relevance, and any
conditional-independence factorization are **not computed by HIA-07**; those
belong to HIA-08 and HIA-09. The response makes this explicit with
`conditional_information_status: not_computed_hia08`.

## Integration and validation

| Artifact | Responsibility |
| --- | --- |
| `structured/hierarchical_level_statistics.py` | Source-pinned level inventory, SU-05 bindings, one-cohort entropy / total correlation and per-node contributions |
| `structured/__init__.py` | Exports `HIA_LEVEL_SCHEMA` and `analyze_hierarchical_level` |
| `ui/workbench_structured_routes.py` | Explicit same-origin POST `/workbench/structured/hierarchy/level-information`, revision check and fresh file verification |
| `ui/static/ml_lab_hia_level.js` | Level selection, manual mappings, opt-in estimator action, report, JSON export |
| `ui/templates/ml_lab_ui/workbench_structured.html` | Level inspector beneath existing graph and preceding HIA panels |
| `ui/static/ml_lab_workbench.css` | Readable panel styles reusing HIA tables |
| `tests/test_hia07_level_statistics.py` | Synthetic redundancy/independence, DAG depth, missingness, repeated values, float binning, source-only, large levels and source-revision gates |
| `tests/test_graph_asset_cache_revision.py` | Keeps the isolated cache-fingerprint test fixture aligned with the HIA-03–07 JavaScript controller inventory |

Run from `~/cluster/ML_Lab`:

```bash
PYTHONPATH=src python -m pytest -q tests/test_hia07_level_statistics.py
node --check src/ml_lab/ui/static/ml_lab_hia_level.js
```

HIA-08 will add conditional information given explicitly mapped preceding
levels, with support and estimator provenance. HIA-09 will add a separate,
explicitly acknowledged conditional-independence approximation and comparison.
