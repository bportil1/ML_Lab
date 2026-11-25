# SU-09 — Matched Structural Retention and Empirical Utility Comparison

**Status:** additive, headless ML_Lab scientific implementation; live PAH/browser acceptance pending. **Depends on:** ML-UW-27–35 and SU-05–08. No shared-renderer, UI, Flask, SQLite, model-training, source-adapter, or input-artifact mutations. No external dependencies or services.

## Scope and estimands

`ml_lab.structured.retention` compares explicitly nominated SU-08 structural candidates against the mandatory `preserve` reference. An exact SHA-256 of the complete original validated `StructuredObject` and the complete SU-05 resolved binding report pin every study and candidate. The evaluator regenerates every candidate via `apply_structural_candidate`; it does not rely on unchecked manifests. Results are detached, deterministic canonical JSON.

**Structural change is not empirical information loss.** Changing nodes, edges, or the layout while retaining all aligned observation columns leaves empirical feature metrics and predictive scores *identical*. Only the explicit `drop_exclusive` SU-08 projection changes the predictor set. An unbound or ambiguously multi-node variable is never silently attributed to a removed node.

Metrics use **the same complete-case observation IDs and one fixed train/test partition for every candidate**. A provided SU-05 group variable can hold potentially dependent siblings entirely together; without a group, callers must explicitly acknowledge assuming independent rows. Neither acknowledgment nor group partition proves independence. Train-only quantile cutpoints and empirical conditional-lookup estimators are fixed across all candidates; unseen feature configurations fall back to training marginals. No training, binning or heldout scoring is performed through navigation or opening the shared graph view.

| Metric | Task | Definition and limitations |
| --- | --- | --- |
| `retained_entropy_fraction` | All | Holdout plug-in joint feature entropy of surviving columns / full feature entropy, using train-fitted numeric bins. Undefined for zero-entropy reference. Not a guarantee of causal or task information preservation. |
| `omitted_conditional_entropy_bits` | All | H(full features \| retained features) estimated on the same holdout, with uncorrected plug-in entropy. |
| `retained_variance_fraction` | Optional, numeric variables | Sum of *holdout* population variances of explicitly requested surviving numeric columns, each scaled by its **training** population variance, divided by same sum for the reference. Not PCA, learned explained variance, or a causal allocation. |
| `retained_numeric_scaled_variance`, `omitted_numeric_scaled_variance` | Optional | Components of the declared scaled variance sum; unavailable if a scale is undefined. |
| `heldout_plugin_mi_bits`, `retained_target_information_fraction` | Categorical target | I(surviving features; target), and its ratio to full-feature I, on the shared holdout. Zero MI denominator is explicitly unavailable. |
| `omitted_conditional_target_information_bits` | Categorical target | I(omitted features; target \| retained features) = I(full; target) − I(retained; target) under the matched empirical distribution. This is associational, not causal. |
| `holdout_accuracy`, `holdout_log_loss_nats` | Categorical target | Smoothed conditional-frequency lookup trained only on training records, with train-only marginal fallback and explicit unseen-target unavailable status. |
| `holdout_mse`, `holdout_r2` | Numeric target | Conditional-mean lookup versus a training-only marginal mean. R² = 1 − model test MSE / constant-model test MSE; undefined for zero denominator. |
| `reconstruction_mse`, `reconstruction_r2` | Target-free with requested numeric reconstruction columns | Exact identity reconstruction when the original numeric column is retained; otherwise train-only conditional-mean lookup from surviving predictors, compared against a train-only marginal-mean baseline. This is an explicit simple estimator, **not a learned inverse of the source graph**. |
| `*_delta_vs_preserve` | Predictive | Candidate minus reference on the *same* holdout; not a statistical significance test. |
| Structural complexity | All | Retained/removed nodes, edges, aligned columns, and cells (= rows × columns) as **count-only proxies**, not measured runtime or actual compressed byte sizes. |

If no target is provided, only the explicitly requested target-free criteria are available; numeric target MI is not silently approximated by discrete binning. Missing/null/unknown/repeated cells are excluded on a **shared complete-case mask** before splitting, with per-variable status diagnostics; repeated source values become one scalar only when the user's SU-05 binding explicitly declares `mean_numeric`. Weight-bearing SU-05 bindings are rejected until a mathematically consistent weighted comparator is implemented. No observations or group rows are inferred from source graph nodes.

Unrequested metrics, joint-state budget violations, zero denominators, and unseen holdout target categories return explicit status and `null` where appropriate, never false zero accuracy or made-up information. A study with insufficient train/test rows, stale fingerprints, or invalid objectives is rejected before scoring. Overflow, stale fingerprints, unknown columns, invalid targets and missing group labels fail closed. Default maximum: 24 candidates, 50,000 complete input rows, 4,096 joint feature states; no silent truncation.

## Auditable thresholds and tradeoffs

A caller can supply an **explicit named** `threshold_metric` and `minimum_threshold`, such as `retained_variance_fraction >= 0.95` or `holdout_accuracy >= 0.95`. This never means an unspecified "95% information preserved." Undefined scores do not qualify. The report also includes a deterministic Pareto list using a specified primary metric (or default task score) and separately minimizing three structural/storage count axes. These are exploratory rankings on **the same holdout**: choosing from them invalidates interpreting those holdout scores as an independent final test. Reevaluate chosen candidates on a new, untouched final set before any performance claim.

## Usage (existing local Python session)

```python
from ml_lab.structured import (
    ObservationBindingPlan, VariableBinding, StructuralCandidatePlan,
    StructuralRetentionStudy, bind_observations, binding_sha256,
    source_sha256, compare_structural_retention,
)

# `source` is an already locally loaded validated StructuredObject.
source_id = source_sha256(source)
binding = ObservationBindingPlan(source_id, source.observation_unit, (
    VariableBinding('feature_a', ('exact-source-node-a',)),
    VariableBinding('feature_b', ('exact-source-node-b',)),
    VariableBinding('outcome', ('exact-target-source-node',)),
), group_variables=('record_cluster',))
bind_id = binding_sha256(bind_observations(source, binding))
plans = (
    StructuralCandidatePlan(source_id, 'preserve', binding_digest=bind_id),
    StructuralCandidatePlan(source_id, 'flatten', binding_digest=bind_id),
    StructuralCandidatePlan(source_id, 'prune', ('exact-source-node-b',),
        observation_policy='drop_exclusive', binding_digest=bind_id),
)
study = StructuralRetentionStudy(
    source_id, bind_id, plans, ('feature_a', 'feature_b'),
    kind='categorical_target', target='outcome', group_variable='record_cluster',
    seed=2026, holdout_fraction=0.25,
    threshold_metric='holdout_accuracy', minimum_threshold=0.80,
)
comparison = compare_structural_retention(source, binding, study)
comparison.to_json()  # archival provenance; no filesystem writes by evaluator
```

For a target-free numeric experiment, use `kind='target_free'`, `target=None`, set `variance_variables=('feature_a',)` and/or `reconstruction_variables=('feature_a',)`, and choose a threshold metric explicitly. The **selected** original target/reconstruction variables are never manufactured from graph topology.

## Verification

```bash
PYTHONPATH=src pytest -q tests/test_su09_structural_retention.py \
    tests/test_su08_structural_candidates.py \
    tests/test_su07_branch_information.py \
    tests/test_su06_branch_uncertainty.py \
    tests/test_su05_structural_observation_binding.py \
    tests/test_ml_uw_35_representation_comparison.py
```

The tests enforce holdout alignment, partition-group isolation, deterministic seeded replay, source/binding version checks, numeric and categorical targets, target-free utility, zero-reference unavailable states, explicit predictor loss, finite estimator budgets, repeated-value and missingness boundaries, and the nonnegotiable **topology-only scoring invariant**. This patch does not claim end-to-end pyPIQUE/PAH browser acceptance, cross-module AST handoff, full-suite acceptance, or a trained task-specific predictive model.
