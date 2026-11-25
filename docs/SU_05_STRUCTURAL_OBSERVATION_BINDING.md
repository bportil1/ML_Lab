# SU-05 — Structural Observation Binding and Sampling Semantics

**Scope:** headless, source-neutral scientific contract. **No UI/Flask/SQLite or graph-renderer changes.** Reuses ML-UW-28 `StructuredObject`, ML-UW-27 observation/cell states, and the SU-04 view without changing the view. SU-06 and SU-07 will use these bindings to compute branch-level descriptive uncertainty and information, respectively.

## Why this matters

A `StructuredObject` can hold both a graph and aligned observation cells, but no rule previously established which observation variable belongs to which original structural node/path. A raw source graph (XML tree, AST, DAG, cyclic graph, etc.) does **not** establish statistical observations, independent samples, or a probabilistic dependency model. A source-specific adapter must provide the data and stable IDs; the core must not guess.

`ml_lab.structured.bindings` introduces a **versioned binding plan** and **validated binding report** with no IO and no domain-specific assumptions.

- Plan schema: `ml-lab.observation-binding-plan@1`.
- Result schema: `ml-lab.observation-binding@1`.
- Source is pinned by a SHA-256 over the **entire canonical validated StructuredObject JSON**, not merely a user-supplied `object_id`. A changed source fails closed.
- Each `VariableBinding(variable_id, node_ids=...)` explicitly links an existing aligned observation column to exact source node IDs. No label/name matching or node-to-sample joins are attempted.
- A variable may explicitly refer to several source nodes only with `merged_sources=True`, in which case the report marks per-node attribution **ambiguous**. It never allocates occurrences to individual nodes.
- For verified `tree`/`forest` topology, the report derives every node's direct and descendant variable-ID sets using validated structural edges. For a DAG, cyclic network, or undirected graph, only declared **direct** node mappings are available; branch descendant scopes are `null` because the schema makes no unique-parent hierarchy assertion.
- Observed rows keep the source's unique `observation_id` and sampling unit. Each observation contributes **one row**, regardless of the number of repeated cell occurrences. Nothing in this stage establishes sampling independence.
- Cell statuses and source null-reason codes (`missing`, `explicit_null`, `empty`, `unknown`, repeated values and adapter-provided reasons) are preserved. Source-specific terms never enter the core algorithm.

## Repeated-value policy

A binding explicitly declares one of the following:

| Policy | Behavior |
| --- | --- |
| `preserve` (default) | Retain the normalized `multiple` cell and every occurrence/lineage entry, without claiming a scalar value |
| `scalar_only` | Reject a repeated cell rather than picking an arbitrary member |
| `mean_numeric` | For repeated values, calculate a per-record arithmetic mean **only if every occurrence is present, finite and numeric**. Reject null/unknown/mixed occurrences rather than silently dropping or imputing. The resolved cell keeps the original multiple cell in its lineage and records the aggregation. Missing/null single cells retain their original statuses. |

These are data-resolution policies **before** SU-06/07 statistics. No variance, MI or CMI is computed in SU-05. Selecting `mean_numeric` does not increase the number of independent samples or claim that mean reduction is lossless.

## Groups, weights and leakage

A plan may designate aligned `group_variables` (one or more observed columns) and an optional `weight_variable` (strictly positive finite numeric). Groups are user/adapter-declared **observation labels**, not inherited from graph parentage. Unsupported/absent group cells are flagged as unavailable, not imputed. Weights are retained in the projected rows and summed for diagnostics but **not automatically applied** to any downstream statistic. Distinct group keys are deterministically encoded; group selection does not establish sampling independence or leak-safe train/test partitions. That is an explicit requirement for later studies.

## API example (illustrative source IDs)

```python
from ml_lab.structured import (
    ObservationBindingPlan, VariableBinding, bind_observations, source_sha256,
)

# source is already a validated StructuredObject with aligned record rows.
plan = ObservationBindingPlan(
    source_digest=source_sha256(source),
    observation_unit=source.observation_unit,
    variables=(
        VariableBinding('temperature', ('/measurements/temperature',)),
        VariableBinding('pressure', ('/measurements/pressure',)),
    ),
    group_variables=('device_id',),
)
bound = bind_observations(source, plan)
report = bound.report  # detached coverage, mapped branch IDs, sampling diagnostics
rows = bound.rows      # exactly one projected row per recorded observation
```

Node identifiers in a binding must match the actual adapter's IDs; the example names are illustrative, not built-in XML knowledge. The mapping is an **explicit declaration** and cannot be certified true merely from a matching string. Source/adapters must provide trustworthy lineage for stronger semantic validation.

## Critical integration limitation and next steps

The existing `XMLStructureAdapter` imports canonical structural **summary** artifacts with `observations=[]`; the `JSONAdapter` imports one JSON **instance** tree, also without an empirical observation population. SU-05 never fabricates record-level values from either. A future source-specific record exporter/adapter needs to pair **real per-record values** with the canonical structural IDs to use empirical branch statistics on actual XML/JSON collections. The existing XML collection exporter may provide this upstream; verify its output/lineage before claiming an end-to-end mapping. Likewise, AST/network source nodes are not samples without an explicitly selected study population.

**SU-06** consumes declared subtree variable sets to calculate descriptive uncertainty/valid variance with clearly specified numerical objective and support. **SU-07** adds branch-level entropy and MI/CMI with declared estimator and conditioning populations. **SU-10** visualizes those reports with the *same PAH_Shared viewer* as pyPIQUE. This patch does not imply those stages are complete.

## Verification gates

- Plans are strict/versioned; duplicate JSON keys, unknown source IDs/variables and stale source fingerprints fail closed.
- Source/observation IDs survive binding; no graph-join row multiplication.
- Repeated elements, exact null source reasons, occurrence order and explicit numeric aggregation are tested.
- Tree and forest descendant binding scopes use original edges; DAG/cyclic graphs do not inherit tree-only claims.
- Optional grouping, sample weights, absent observations and larger/deeper trees are tested.
- Source object remains immutable; result reports and resolved rows are detached JSON-compatible values.
- No source-domain terms hardcoded in the scientific core; no network access, filesystem I/O, migrations or automatic execution.
