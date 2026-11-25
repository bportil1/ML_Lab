# SU-07 — Empirical Branch Information, Target Relevance and Redundancy

**Scope:** headless, source-neutral scientific analysis of SU-05 bindings. **Dependency:** SU-05 (binding contract), SU-06 (fingerprint helper); reuses existing ML-UW-32 discrete information primitives. **No** graph renderer, Flask route, SQLite, transformation, or model semantics changed. Real UI/PAH-hosted acceptance remains for SU-10.

## Purpose and meaning

`ml_lab.structured.branch_information` reports finite-sample **empirical discrete** information over *aligned observation rows*. A structural branch is only a declared set of variables: it is **not** a sample, inference graph, causal direction, or probabilistic dependency.

The source must first be validated as a `StructuredObject`, then passed through a source-digest-pinned `ObservationBindingPlan`, producing an immutable SU-05 `ObservationBindingResult`. The SU-07 plan pins the **entire binding result** by SHA-256. Modifying the underlying observations, binding plan or source invalidates that digest.

For each eligible selected source node, let `S` be the non-ambiguous source-scoped observed variables with an explicitly declared target `Y` excluded. Let `R` be the user-declared **retained predictor** set. `B = S \ R` is the **incremental candidate branch**; a branch with `B = ∅` is explicitly unavailable rather than spuriously reporting zero information.

The following separately labeled quantities are computed **over a common complete-case cohort for each node**:

| Output | Meaning | Decision caveat |
| --- | --- | --- |
| `entropy.scope_joint` | Joint entropy `H(S)` | Includes in-scope retained variables; do not confuse with novel candidate entropy |
| `entropy.candidate_joint` | Joint entropy `H(B)` | **Unsupervised descriptive** variability, not a prediction score |
| `entropy.candidate_given_retained` | `H(B \| R)` | Novel unsupervised variation conditional on retained variables; equals `H(B)` if `R` empty |
| `relevance.target_scope_mi` | `I(Y; S)` | Branch-wide target relevance, including any retained predictors already in scope |
| `relevance.target_candidate_mi` | `I(Y; B)` | Marginal relevance of only the novel candidate features |
| `relevance.target_candidate_cmi_given_retained` | **`I(Y; B \| R)`** | Incremental relevance **after retained variables**; equals marginal relevance when `R` empty |
| `relevance.target_entropy_given_retained` | `H(Y \| R)` | Target uncertainty baseline under declared retained context |
| `redundancy_candidate_retained_mi` | `I(B; R)` | Statistical redundancy; unavailable rather than numeric zero if `R` is empty |

**A low-entropy branch can be rare but informative. A feature with zero marginal MI can have high conditional MI (e.g., XOR). No reported MI/CMI or redundancy number by itself authorizes pruning.** Predictive held-out utility and structural candidate loss are separately addressed in SU-08/SU-09.

## API

```python
from ml_lab.structured import (
    ObservationBindingPlan, VariableBinding, source_sha256, bind_observations,
    binding_sha256, BranchInformationPlan, analyze_branch_information,
)

binding = bind_observations(source, ObservationBindingPlan(
    source_sha256(source), source.observation_unit,
    (VariableBinding('feature_a', ('source-node-a',)),
     VariableBinding('feature_b', ('source-node-b',)),
     VariableBinding('label', ('source-label-node',))),
))
plan = BranchInformationPlan(
    binding_sha256(binding), nodes=('source-node-b',),
    target='label', retained_variables=('feature_a',),
    bins=(('feature_b', 'quantile', 4),),  # only if feature_b is numeric/continuous
    missing_policy='complete_case', min_samples=10, min_stratum_samples=3,
)
report = analyze_branch_information(binding, plan).to_dict()
```

The plan schema is `ml-lab.branch-information-plan@1`; the report schema is `ml-lab.branch-information@1`. `BranchInformationPlan.from_json(...)` rejects duplicate JSON keys, fields from a different schema, and malformed discretizers. `nodes=()` means every node in the SU-05 binding inventory within explicit cost bounds; for large structures select source IDs to inspect. Nodes and variables have exact IDs, never label-matched.

## Sample, missingness and discretization semantics

- **Observation:** one SU-05 aligned row = one unit. Repeated cell occurrences are *never additional independent rows*. `multiple` cells used in a requested analysis cause an explicit error; resolve them through SU-05 (`scalar_only` or `mean_numeric`) first. Merged source-node identities are excluded from node attribution with an explicit `ambiguous_merged_variable_ids` list.
- **Weights:** empirical-discrete estimator is currently **unweighted**. A non-unit SU-05 weight fails closed; it is never silently discarded, replicated, or treated as extra evidence. A true weighted estimator would require a new explicit, separately validated policy.
- **Missingness:** `complete_case` (default) excludes each row with `missing`, `explicit_null`, or `unknown` in the relevant branch/target/retained variables; `as_category` uses distinct missing-state categories without source-reason imputation; `error` rejects them. `empty` is a separate encoded state. Reported exclusions and included rows are per node and per analysis cohort. Results from different nodes can use different cohorts and therefore are not automatically comparable.
- **Numeric data:** integers and booleans are discrete categories by default. Floating values require *explicit* equal-width or quantile binning. Integer values may also be explicitly binned. Binning uses the existing ML-UW-32 conventions; bin definitions are calculated **once from all available aligned observations**, not silently refit per branch. They are **descriptive**, **not** fitted on a leakage-safe train split. Any SU-09 predictive evaluation must refit on training data only or reuse appropriately split-fitted bins.
- **Estimand:** uncorrected empirical-discrete plug-in estimates in bits by default. No Bayesian CPT fitting, conditional independence test, p-value, variance attribution, causality, generalization guarantee or confidence interval. Joint feature vectors are jointly encoded rather than summed from marginal MI contributions.
- **Support:** minimum overall sample count and minimum conditioning-stratum count are explicit. When any conditioning stratum fails the threshold, conditional estimates return `insufficient_conditioning_support`, while valid unconditional metrics can still be reported. Includes joint cardinality, singleton states, conditioning strata, missingness/exclusion diagnostics, and finite-sample bias warnings.

## Graph presentation contract

The returned `association_layer` has schema `ml-lab.statistical-node-overlay@1`: source node ID, node-scoped information metric envelopes (`status`, `value`), with **empty** `source_edges` and `probabilistic_edges`. This layer is distinct from source structure, independent of the pyPIQUE probabilistic network, and has `causal_interpretation=False`. SU-10 can attach overlays to the existing PAH_Shared viewer without inventing any source-to-target edges. For trees/forests, SU-05-defined subtree variable membership is honored; for DAGs, cyclic graphs and other nonhierarchies, only explicit *direct* node mappings are reported.

## Limits and acceptance

Analysis fails visibly rather than silently sub-sampling if there are over 50,000 aligned rows, over 256 selected nodes, or over 5,000,000 node × row comparisons. Nodes with over 12 joint variables have `joint_dimensionality_limit`. An attempted estimate with more than 5,000 observed joint states has `joint_state_limit`. No pruning decisions or statistical edges are generated.

Focused regression fixtures cover XOR synergy (`I(Y; B)=0`, `I(Y; B | A)=1`), exact redundant clones, rare informative binary signals, multivariable joint entropy, missingness strata, fixed discretization, repeated cells, source-identity ambiguity, nonhierarchical direct-only evidence, and fingerprint/serialization safety. Previous SU-05, SU-06, ML-UW-32, and descriptive uncertainty tests should still pass.

```bash
PYTHONPATH=src pytest -q tests/test_su07_branch_information.py
```
