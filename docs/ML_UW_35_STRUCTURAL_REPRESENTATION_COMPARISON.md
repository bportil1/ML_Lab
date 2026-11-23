# ML-UW-35 — Matched Structural Representation Comparison

Status: headless scientific API, no Flask/PAH UI integration or SQLite migration.
Depends on: ML-UW-27 through ML-UW-34.

## Purpose and boundaries

ML-UW-35 introduces a deterministic, domain-neutral **study plan** and comparison
report for transformed `StructuredObject` views. It answers two different
questions without conflating them:

1. **Structural preservation:** Which node/edge identities, topology family,
   attributes, and observation variables survive `full`, `depth`, `aggregate`,
   `subtree`, or `flat` transformations? The ML-UW-30 transformation audit is the
   authority; the report adds retention ratios and view serialized-byte counts.
2. **Observed predictive/information utility:** What happens to a **predeclared**
   categorical target when a researcher explicitly retains different subsets
   of **aligned observation variables**? All candidates use the exact same
   complete-case observation population and a single held-out partition.

**Crucial:** ML-UW-30 structural operations do not derive graph features from
nodes or links. Consequently, merely flattening a graph while retaining all
observation features **must not change predictive utility**. Graph topology
itself cannot be assigned a predictive benefit/loss until an explicit,
validated topology-to-feature extraction procedure exists. This increment
neither infers graph dependencies nor implements representation learning.

## Public API

- `ComparisonCandidate(label, plan: TransformationPlan, predictors: tuple[str, ...])`
- `ComparisonStudy(target, candidates, group_variable=..., seed=...,
  holdout_fraction=..., pseudocount=..., min_train_rows=...,
  min_test_rows=..., max_joint_states=...)`
- `compare_representations(source: StructuredObject, study: ComparisonStudy) -> dict`
- `ComparisonStudy.to_json()`, `.to_dict()` and `.from_dict()` for deterministic
  plan exchange.

A candidate named `full` is required as an unprojected `TransformationPlan('full')`
reference. Additional candidates may use any ML-UW-30 transformation. Candidate
predictors must be subsets of the reference predictors; the target is never a
predictor. An explicit `plan.variables` projection must retain the target and
all declared candidate predictors. A requested structure transformation may
only be used with an input satisfying the corresponding topology capability.

### Example (pseudocode; `source` is a validated StructuredObject)

```python
from ml_lab.structured import (
    ComparisonCandidate, ComparisonStudy, TransformationPlan,
    compare_representations,
)

study = ComparisonStudy(
    target='outcome',
    group_variable='source_project',  # partition entire projects together
    candidates=(
        ComparisonCandidate('full', TransformationPlan('full'), ('a', 'b')),
        ComparisonCandidate('partial', TransformationPlan(
            'depth', max_depth=2, variables=('outcome', 'a')), ('a',)),
        ComparisonCandidate('flat', TransformationPlan('flat'), ('a', 'b')),
    ),
    seed=11,
    holdout_fraction=0.25,
    pseudocount=1.0,
)
report = compare_representations(source, study)
```

The grouping column can be dropped from a candidate's projected output because
splitting uses the original source observation rows, **before** transformations.
If no grouping column is provided, the caller must set
`acknowledge_independent_rows=True`. This is not a statistical independence
certificate; hierarchical samples should use a stable source-group variable to
keep sibling/nested observations together.

## Statistical contract

- **Observation units:** the source's existing observation IDs, never graph
  nodes or edges interpreted as independent samples. No node–observation joins.
- **Common support:** incomplete target, group, or **any candidate predictor**
  removes that row from *all* candidates, with excluded counts reported.
  `missing`, `explicit_null`, `unknown`, and `multiple` are not imputed.
- **Data types:** scalar categorical strings, integers, booleans, and explicit
  empty-string states. Continuous floats, arrays, dictionaries, and repeated
  values require a separately documented preprocessing/aggregation step.
  This version does not bin floats automatically or estimate continuous MI.
- **Holdout:** deterministic seeded hash partition on unique group identities;
  all siblings of the same declared group stay on the same side. Without a
  group variable, use a deliberately acknowledged row partition. The report
  includes train/test observation IDs, partition sizes, seed, and SHA-256 ID
  digests. At least one group is retained on each side.
- **Prediction:** transparent conditional-frequency estimator of categorical
  `P(target | predictors)` using only the training partition, with symmetric
  positive pseudocounts. For unseen feature configurations, use the *training*
  smoothed marginal distribution (reported as a fallback). Unknown target
  states appearing in holdout are explicitly unscorable; accuracy and log loss
  become `None` rather than silently discarding those observations.
- **Evaluation:** matched holdout mean negative log-likelihood in **nats**
  (lower is better) and accuracy (higher is better). The scorer is an explicitly
  documented discrete prediction baseline, *not* the general ML-UW-34 BN
  inference engine or an automatically selected probabilistic graph.
- **Information:** the existing ML-UW-32 empirical discrete information
  estimator is reused on training-only categorical joint-state encodings.
  Report `H(target)`, `I(target; retained predictors)`, and, for omitted
  predictors, `I(target; omitted | retained)`. Comparisons always retain the
  **same target and sample support**. No entropy is subtracted across changing
  variable spaces to imply a meaningful loss. Unsupported conditioning strata
  produce `status` and `None`, not a fabricated zero. Identity omissions have
  exactly zero incremental information by definition.
- **Statistical caveats:** in-sample plug-in information measures have finite-
  sample bias; no independence tests, p-values, confidence intervals, or
  causal claims are produced. Outcome-informed candidate selection based on
  the same holdout scores requires a subsequent independent final test.

## Report and resource budgets

The report includes a versioned schema, source fingerprint, study plan, common
support, single split, per-candidate transformation audits and retention,
training-only information results, holdout prediction results, relative holdout
loss to full, and a clearly labeled advisory. A SHA-256 study digest covers all
fields, and repeated inputs/configurations are deterministic.

`view_json_bytes` is a serialized artifact-size proxy; **execution time, peak
memory, reconstruction fidelity and downstream probabilistic-model comparison
are not measured**. In particular a depth reduction recording removed node
identities does not constitute a quantitative reconstruction-error estimator.

Limits: at most 12 candidates, 50,000 source observations, 4,096 distinct
training feature configurations (configurable downwards), and bounded minimum
train/holdout support. Exceeding caps fails explicitly, with no silent sampling.

The core neither reads/writes local files nor shares module SQLite databases.
There is no automatic computation on UI navigation. The eventual ML-UW-36 UI
should make input readiness and the distinction between source topology and
feature projection clear; ML-UW-37 adapter pilots should use exported data or
module APIs.

## Acceptance scope

Tests exercise deterministic common splits, group-disjoint sibling holdouts,
explicit null exclusions, XOR-like conditional information, withheld-feature
predictive utility, equality of full/flat predictive outcomes when features are
unchanged, alias-safe reserved names, sparse conditioning support, unseen
holdout target labels, objective limits, input validation, and no IO side
-effects. Existing ML-UW tests must remain green.

Deferred to later research: learned topology representations, inference over
source edges, continuous-density information estimation, uncertainty
intervals/bootstrap with group-block resampling, and arbitrary Bayesian-network
holdout scoring. Nothing in this report claims those analyses are implemented.
