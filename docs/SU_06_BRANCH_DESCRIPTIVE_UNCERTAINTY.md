# SU-06 — Branch Descriptive Uncertainty and Variance

**Status:** scientific-core patch. Requires SU-05. No Flask, browser, database, graph renderer, trained models or automatic execution is introduced. Real-data and PAH-hosted acceptance remain separate.

## Purpose and evidence boundaries

This increment describes **actual aligned observations** linked in SU-05 to stable original source IDs. It does not turn XML nodes, AST nodes, tree edges, repeated elements, or source paths into extra samples. It adds one scientific service, `ml_lab.structured.branch_uncertainty`, and two strict versioned contracts:

- `ml-lab.branch-uncertainty-plan@1`: pinned SU-05 binding SHA-256, numeric targets, optional ordered observed-group prefix, scale, objective and positive weights;
- `ml-lab.branch-uncertainty@1`: detached, JSON-compatible empirical evidence and original-ID overlay references.

The binding digest is over the *complete serialized SU-05 result*, including the plan and bound rows. The source digest is carried through from SU-05. Changed observation values or source structure invalidate the analysis plan. There is no filesystem/network access or implicit data loading.

## Public Python interface

```python
from ml_lab.structured import (
    BranchUncertaintyPlan, analyze_branch_uncertainty, binding_sha256,
)

# SU-05 must already have produced `binding` from an explicitly validated
# StructuredObject and ObservationBindingPlan.
plan = BranchUncertaintyPlan(
    binding_digest=binding_sha256(binding),
    targets=('latency', 'power'),
    group_levels=('project', 'module'),  # a PREFIX of SU-05 group_variables
    scale='declared',
    declared_scales=(('latency', 100.0), ('power', 10.0)),
    objective='weighted_variance',
    target_weights=(('latency', 3.0), ('power', 1.0)),
)
report = analyze_branch_uncertainty(binding, plan).to_dict()
```

`group_levels` refer to explicitly provided **observation labels**, not to structural parents or children. They must be an ordered prefix of SU-05 `group_variables`, to ensure a valid nested partition. Duplicate IDs and unbound targets fail closed. The report never asserts that samples are independent, that groups are random effects, or that graph edges imply variance causation.

## Statistics and denominators

For **each mapped variable**, on all bound records, the report includes:

- Original and resolved cell-status distributions, explicit missing/null reason counts, and repeated-occurrence status and null-reason counts.
- Observed-record occurrence rate, original per-record occurrence cardinality (including zeros), total occurrences and present occurrences. Repeat occurrences **do not change record counts**.
- Number of distinct observed value encodings, exact up to 5,000 values; above that limit, the value is `null` and status `above_limit` (never a fabricated exact count).
- The original/source ID correspondence for both direct and hierarchical subtree references. Many-to-one source-node mappings remain flagged `ambiguous_merged_variable_ids` and are excluded from the node-level selected-target overlay.

**For each requested numeric target**, all targets use the same complete-case row cohort. Rows are included if all target values are present, finite numbers and every requested grouping label is available. Missing labels/outcomes are excluded once per record with a deterministic recorded reason. Unrequested group variables do **not** cause exclusions. `mean_numeric` SU-05 resolved values are numeric inputs; original occurrences are retained solely as separate descriptive cardinality evidence. A `preserve` repeated-valued target is unavailable for scalar variance, not silently averaged.

Each requested grouping level also reports complete-case group support and weighted totals (capped at 2,000 displayed identities per level, with explicit truncation); group-level summaries are not inferred from the original source topology.

Numeric summaries report count, min/max, weighted mean, weighted-population variance and SD, weight total and Kish descriptive effective count, plus **unweighted sample variance** where at least two real rows exist. The *population* denominator is the sum of observation weights; these are descriptive positive analysis weights rather than a claim of iid or frequency-sampling corrections. Variance for fewer than two records is explicitly `null`. Constant numerical targets have zero raw variance with status `constant`.

## Objectives and normalization

- `objective='per_target'` (default): report each requested numeric target separately. Multiple targets can retain original units because their variances are **not** summed.
- `objective='weighted_variance'`: requires a positive target weight for **every** target. For more than one target, `scale='raw'` is rejected. Either declare positive per-target unit scales (`scale='declared'`) or use the empirical weighted complete-case SD (`scale='zscore'`). Both operations are descriptive; the latter is in-sample normalization, not a fitted predictive transformation.
- `objective='reconstruction_mse'`: records an explicit future reconstruction intent but returns `status='unavailable', reason='paired_reconstruction_required'`. SU-05 contains no paired candidate/reconstruction values, and SU-06 will not invent them. A later comparison sprint can calculate reconstruction loss only after supplying verified paired predictions.

The weighted objective is a **normalized weighted average of scaled target population variances** over the same complete-case cohort. In explicit nested groups, total weighted sum-of-squares is partitioned into between-group contributions at each declared level and within-leaf contributions. The combined objective partition uses the corresponding weighted population variance per target, then weighted averaging. Unbalanced groups and fractional observation weights are supported. Fractions are `null` for zero total dispersion.

If normalization is impossible (zero standard deviation or no sufficient support), the affected scaled summary is `null`, and a multi-target weighted objective reports `unavailable` rather than a misleading zero. No confidence intervals, statistical tests, Bayesian uncertainty, imputation, independence estimation, leakage assurance or causal effect sizes are computed.

## Graph overlay contract

`nodes[]` contains `node_id` from the original SU-05 structure, `scope` (`subtree` for verified hierarchy, `direct_only` for DAG/general graphs), `direct_variable_ids`, `scope_variable_ids`, `selected_target_ids`, `ambiguous_merged_variable_ids` and an explicit `variance_is_attributed_to_graph_edge: false` sentinel. Consumers can join `selected_target_ids` to `targets[variable_id]` for read-only inspection and overlay design, but **must not sum or color nodes as an additive explained-variance allocation**: ancestor and child scopes can overlap, and one variable can be referenced by many ancestors.

No viewer implementation or overlay rendering changes are made in SU-06. The existing SU-04 PAH_Shared graph renderer remains the single renderer owner.

## Validation

```bash
PYTHONPATH=src pytest -q tests/test_su06_branch_uncertainty.py
PYTHONPATH=src pytest -q tests/test_su05_structural_observation_binding.py tests/test_ml_uw_31_descriptive_uncertainty.py
```

Acceptance fixtures cover unbalanced nested groups, fractional observation weights, null/missing and repeated observations, immutable source and detached reports, exact node IDs and non-hierarchy direct-only mapping, merged-node ambiguity, zero-support and constant signals, explicit scaling, objective accounting, stale SHA-256 references, plan schema validation and complete-case exclusions.

**Not addressed by SU-06:** candidate transformations, retention/utility scoring, information theory/MI/CMI (SU-07), interactive overlay controls, new source adapters and production PAH integration.
