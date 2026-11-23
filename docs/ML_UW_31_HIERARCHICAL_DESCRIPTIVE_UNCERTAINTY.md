# ML-UW-31 — Domain-neutral descriptive structure and uncertainty (Option A)

**Baseline:** `ML_Lab-main(5).zip` + ML-UW-28, 29, 30. **Scope:** new side-effect-free scientific module `ml_lab.structured.descriptive`; imports are exposed in `ml_lab.structured`. No adapter, existing XML parser, SQLite schema, PAH route, or UI behavior changes.

## Four validated entry points

```python
from ml_lab.structured import (
    describe_structure, describe_collection, describe_observations,
    describe_nested_variance,
)

# A single source: exact topology and raw cardinalities; NO guessed occurrence probabilities.
shape = describe_structure(obj)

# Multiple sources: caller explicitly asserts that IDs match across objects.
coverage = describe_collection([obj1, obj2], node_identity='id', edge_identity='id')

# Actual aligned observations (not the graph nodes): counts, reasons, rates, numeric stats.
values = describe_observations(obj, variables=('v', 'group'), numeric_variables=('v',))

# Explicit nesting of grouping variables, not inferred from structural edges.
variance = describe_nested_variance(
    obj, value_variable='v', group_levels=('project', 'module'),
)
```

All reports are detached, JSON-compatible, versioned objects with explicit input IDs, denominator/support counts, and limitations. Invalid input is rejected using `StructuredContractError` instead of silently coercing data.

## Scientific contracts and availability

| Calculation | Required evidence | Returned meaning | Not inferred |
| --- | --- | --- | --- |
| One-object node/edge counts, degree | Validated `StructuredObject` | Counts, node/edge kinds and degrees | Samples, probabilities, dependence |
| Child cardinality, roots, depth, leaves | Verified rooted tree/forest capabilities | Outgoing edge count per parent (leaves included); depth distributions | Probabilistic parent-child association |
| Cross-object path/edge presence rates | Explicit identity selectors and one or more validated objects | `present_objects / total_objects` with absent counts and within-source multiplicity | Semantic node alignment or independence |
| Missingness, explicit-null/unknown/empty statuses | Named aligned observation variables | Per-variable status counts/rates over **observation rows**, with source-supplied reason strings | XML assumptions, imputation, duplicate child samples |
| Numeric value summaries | Explicit numeric variables and finite observed numeric values | Mean, min, max, sample variance when at least two observations | Variance of insufficient data, automatic aggregating of `multiple` |
| Nested within/between variability | Explicit numeric outcome + ordered aligned grouping variables | Exact descriptive sum-of-squares partition into each nested level and residual within deepest groups | A hierarchical source-graph factorization, random effects, uncertainty intervals, causality |

**Collection identity rules:** `node_identity` and `edge_identity` must be explicit selectors (`id`, `attributes:<key>` or `metadata:<key>`). These are *assertions* that equal keys denote comparable entities across objects. An absent/non-string identity raises an error. The engine never aligns similar-looking labels, traverses source files, or assumes that a source's node IDs represent compatible concepts. Repeated identities within one object contribute one presence, with their total multiplicity tracked separately. Identical-content objects intentionally count as separate supplied objects; independence has not been established.

**Single-object structural metrics are not cross-source uncertainty estimates.** Even if the source is an aggregate XML structure artifact, generic structural node counts alone do not establish source-level path probabilities. Domain adapters must provide explicit aligned identities or observations.

**Null policy:** A missing, explicit-null or unknown aligned cell contributes one observation to the status denominator and zero numeric samples; its adapter-provided `source_reason` is counted opaquely. `empty` is not a null and is counted as observed, but is not automatically parsed into a number. `multiple` is counted as one parent observation; nested occurrence status/reasons are reported separately without exploding the sample count. Non-numeric, boolean and non-finite present numeric values raise errors. Numeric variance requires `n >= 2`; no evidence is represented by `None`, not a fabricated zero.

## Hierarchical variance decomposition

For complete-case, **aligned observation rows** with numeric outcome `Y` and explicitly ordered group labels `(G1, ..., Gk)`, define mean `m` over N included rows. The observed descriptive sum of squares is:

`SS_total = sum_i (Y_i - m)^2`

At each grouping level `j`, for each group `g` of that level with parent prefix `parent(g)`, compute `n_g * (m_g - m_parent(g))^2`. At the leaf level, calculate `SS_within = sum_i (Y_i - m_leaf(i))^2`. Thus `SS_total = sum_j SS_level(j) + SS_within` up to floating point precision, including for **unbalanced groups**. The grouping keys always include the full parent-prefix, so identically named child groups under different parents are separate. Normalized proportions are returned only if `SS_total > 0`. There is no random-effects interpretation or hypothesis testing in this sprint.

Outcome nulls and missing group labels are excluded from the complete-case variance decomposition with exact counts/reasons; if fewer than two complete rows remain, SS components and sample variance are unavailable (`None`). At least two observations with constant Y produce real zero total variability, with undefined normalized proportions. Overflow is rejected rather than producing NaN/Infinity.

**Critical distinction:** the source graph's parent-child edges are *structural*, not observations about a probability distribution. Nested group levels are caller-supplied aligned variables. A DAG/cycle can have structural degree summaries but no tree-only root/depth report. These descriptive quantities do not imply any Option B MI or Option C probabilistic graph.

## Interface and acceptance boundaries

Use this as a library API in ML-UW-31. The intended UI integration into Feature Profiling and cross-representation comparison remains ML-UW-36. No background compute, caches, automatic source loading, data mutation, or training are initiated.

```bash
PYTHONPATH=src pytest -q \
  tests/test_ml_uw_27_structured_contract.py \
  tests/test_ml_uw_28_structured_object.py \
  tests/test_ml_uw_29_source_adapters.py \
  tests/test_ml_uw_30_structural_transformations.py \
  tests/test_ml_uw_31_descriptive_uncertainty.py
```

Further work: ML-UW-32 adds mathematically specified information-theoretic estimators using **paired aligned** variables and documented missingness/estimation policies. ML-UW-33/34 adds explicit probabilistic graph assumptions and inference. Neither is included here. Flask/PAH deployment acceptance is separate from headless tests.
