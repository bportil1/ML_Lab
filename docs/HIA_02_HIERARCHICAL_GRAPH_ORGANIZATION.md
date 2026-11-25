# HIA-02 — Deterministic source hierarchy and scope membership

**Status:** Source-level, additive implementation; no new UI or statistical estimator.

**Owner:** standalone `ML_Lab` repository. The input is the existing immutable
`StructuredObject` from the ML-UW-28 scientific contract; the optional
observation inventory is the source-pinned SU-05 `ObservationBindingResult`.
Nothing is written into the source, PAH or another repository.

## 1. What this patch actually computes

`ml_lab.structured.hierarchical_index` provides:

- `build_hierarchy_index(source, level_policy=None)`: complete source-pinned
  index of roots, deterministic level memberships, direct source-edge parents
  and children, and exact original node and edge identities. There is no
  node-count, edge-count, or depth cutoff.
- `HierarchyIndex`: immutable read-only index with `level_of`, `level_ids`,
  `parents_of`, `children_of`, `ancestors_of`, `descendants_of`, `roots_for`,
  `branch_ids`, `prior_level_ids`, `scope_ids` and `node_context`.
  Transitive traversal is **on demand**, iterative, and de-duplicated, not a
  precomputed potentially quadratic all-pairs reachability matrix.
- `inspect_hierarchical_selection(source, request, binding=None, index=None)`:
  resolve an HIA-01 node/branch/level/network selection to original source
  node IDs, identify separately bound SU-05 observation variables, and disclose
  ambiguous merged-node and cross-level variable attribution. This returns
  `ml-lab.hierarchical-scope-selection@1`; it reports *no* MI, CMI, variance,
  entropy, or pruning recommendation.

The inventory returned by `HierarchyIndex.to_dict()` uses versioned schema
`ml-lab.hierarchical-level-index@1` and exposes all original source edge IDs;
it is a **derived structural index**, not a replacement for the original edge
records, node attributes, or source topology. The source itself remains
untouched.

## 2. Exact level semantics

| Source topology | Required policy | Definition | Status |
| --- | --- | --- | --- |
| `tree` | `rooted_depth` | Unique-parent directed distance from the one verified root; root is depth 0 | Supported |
| `forest` | `rooted_depth` | Unique-parent distance from each component's verified root; every root at depth 0 | Supported |
| `dag` | `dag_longest_path` | Longest *directed* path from any indegree-zero root. Multiple parents and disconnected components are retained | Supported |
| `directed_graph`, `undirected_graph` | None | No certified structural-level semantics, even if a directed graph happens to be acyclic | Rejected for HIA levels |
| Source with a directed cycle claiming `tree`/`dag` | None | Invalid topology | Rejected by existing `StructuredObject` validation |

Levels are deterministic under node and edge list ordering: node identities
are sorted for level presentation and adjacency. A DAG edge may jump across
several levels; no edge is rewritten, dropped, split or assigned a probabilistic
parent role. For a selected DAG node `roots_for()` reports **all** reachable
source roots, without assuming unique parentage.

The index has O(V + E) traversal after deterministic sorting of the IDs and
adjacency lists, and O(V + E) memory for the index. It does **not** materialize
one ancestor list per node or one variable-count table per branch.
A single ancestry/descendant query costs O(V + E) in the worst case. Returning
a level's or branch's full membership necessarily costs at least the size of
that result. Huge full exports remain subject to machine RAM, not an arbitrary
scientific graph-count cutoff.

## 3. Scope and observation mapping

The `inspect_hierarchical_selection` report is pinned to the complete
`source_sha256(source)` and, when supplied, the `binding_sha256(binding)`
from HIA-01. A mismatched source, binding or hierarchy index fails closed.

- **Node:** just that exact original node ID.
- **Branch:** node plus unique-parent descendants, only for verified
  `tree`/`forest`. DAG branch *reachability* can be inspected, but HIA
  branch scope is rejected because overlapping DAG descendants would lead to
  ambiguous subtree attribution.
- **Level:** all original nodes at exactly one verified depth.
- **Network:** all original nodes in deterministic level order.
- **Prior levels:** all nodes in earlier depths, supplied as an explicit
  structural conditioning *selection*, **not** as a fitted probability model.

Variables come from explicitly mapped SU-05 binding entries, **never** from
source labels, graph connectivity, or node counts. A variable mapped to two
nodes is disclosed in `ambiguous_variable_ids` rather than counted as two
samples. A variable spanning multiple levels appears in
`cross_level_binding_variable_ids`; it must not be used for additive
level attribution without an explicit resolution policy in later HIA patches.
No actual sample size can be inferred from graph counts. Exact aligned
observation rows are still required for future estimators.

The HIA-01 eligibility gate remains unchanged for backwards compatibility.
HIA-02's selection report updates the **reason** for successfully resolved
levels to `HIA_scientific_estimator_not_implemented` and flags levels with
no mapped observations as `unavailable`. Every numerical metric still has
`estimate: null` and `estimator_support: null`.

## 4. Example

```python
from ml_lab.structured import (
    HierarchicalScope, HierarchicalAnalysisRequest, source_sha256,
    build_hierarchy_index, inspect_hierarchical_selection,
)

# source: an existing validated StructuredObject
index = build_hierarchy_index(source)
print(index.roots, index.levels)
print(index.node_context('an-original-node-id'))

req = HierarchicalAnalysisRequest(
    source_digest=source_sha256(source),
    scope=HierarchicalScope('level', level=2),
    metrics=('level_joint_entropy',),
    level_policy=index.level_policy,
)
report = inspect_hierarchical_selection(source, req, index=index)
# Source node IDs are real; entropy is not yet computed.
assert all(metric['estimate'] is None for metric in report['metrics'])
```

## 5. HIA-03–12 handoff

- **HIA-03:** consume exact node/branch/level identity lists in the existing
  PAH_Shared viewer, preserving progressive graph navigation and exports.
  Structural edges must remain distinct from later optional statistical overlays.
- **HIA-04–05:** node-level observation distributions and neighbor associations;
  the index's `parents_of` and `children_of` describe **only structural**
  neighbors until paired measured variables and estimator provenance exist.
- **HIA-06–09:** add branch/level joint and conditional statistics on verified
  aligned variables, with common cohort, factorization consent and duplicate
  variable policy. Never sum node entropies to claim a level joint entropy.
- **HIA-10–12:** global accounting, overlays, large-graph testing and live
  hosted-browser acceptance.

### Local checks

```bash
PYTHONPATH=src python -m pytest -q tests/test_hia01_hierarchical_contracts.py tests/test_hia02_hierarchical_index.py
```

No UI, database, browser state, graph rendering, source parser, estimation
algorithm, or standalone repo other than ML_Lab is changed by this patch.
