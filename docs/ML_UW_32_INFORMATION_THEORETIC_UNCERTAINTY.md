# ML-UW-32 — Domain-neutral information-theoretic uncertainty

**Status:** headless scientific API implemented. This is Option B of the agreed
hierarchical uncertainty roadmap. UI wiring is reserved for ML-UW-36; graphical
model fitting and inference are reserved for ML-UW-33/34.

## Scientific boundary

All functions take a validated `StructuredObject` from ML-UW-28 and operate
**only on explicitly paired `observations`**. Source node counts, tree depth,
XML elements, AST nodes, and pyPIQUE edges are not statistical sample rows. A
source edge is *never* implicitly treated as a conditional dependence or a
probabilistic factor. Adapters retain source provenance; no domain-specific
terms or database access appear in this estimator.

Mathematical quantities (all with the chosen logarithm base):

- Entropy: `H(X) = -Σ p(x) log p(x)`.
- Conditional entropy: `H(X|Z) = -Σ p(x,z) log p(x|z)`.
- Mutual information: `I(X;Y) = Σ p(x,y) log [p(x,y)/(p(x)p(y))]`.
- Conditional mutual information:
  `I(X;Y|Z) = Σ p(x,y,z) log [p(x,y|z)/(p(x|z)p(y|z))]`.

Here `Z` may be a tuple of multiple **aligned observation variables**.
Estimates use empirical multinomial frequencies. This is a discrete plug-in
estimator, without Miller–Madow correction, p-values, learned dependencies,
or confidence intervals. At low support, especially with many joint categories,
MI and CMI may be substantially positively biased even for truly independent
variables. Reports explicitly disclose this limitation.

## Python API

```python
from ml_lab.structured import (
    estimate_entropy,
    estimate_conditional_entropy,
    estimate_mutual_information,
    estimate_conditional_mutual_information,
)

# source: validated StructuredObject whose observations align these variables
h = estimate_entropy(source, 'module')
mi = estimate_mutual_information(source, 'parent_metric', 'child_metric')
cmi = estimate_conditional_mutual_information(
    source, 'child_metric', 'outcome', ('module',),
    missing_policy='complete_case', min_samples=10,
    min_stratum_samples=2,
)
# Continuous floats require a deliberate variable-specific discretization:
mi_binned = estimate_mutual_information(
    source, 'numeric_measure', 'outcome',
    bins={'numeric_measure': {'mode': 'quantile', 'bins': 4}},
)
```

The underlying `estimate_information(source, *, measure, x, y=None,
given=(), ...)` supports the same operations. The `estimate_*` wrappers return
JSON-safe **reports**, not raw scalar results. Read `report['value']` only when
`report['status'] == 'ok'`.

### Explicit estimator choices

- `estimator='empirical_discrete'` is the only implemented estimator. Other
  algorithms are rejected rather than silently substituted.
- Integer, string and Boolean observations are discrete categories by default;
  types remain distinct (`1`, `True`, and `'1'` are not identical categories).
- Floating-point values require an explicit `bins` entry. Numeric integers may
  also be deliberately binned. Never infer that rounded floats are naturally
  categorical or that source graph connectivity provides a sample distribution.
- Per-variable discretization modes are `quantile` (ties stay in one bin)
  and `equal_width` (minimum-to-maximum bins). Bin counts must be 2–128.
  Reports include requested/realized bin counts and numeric boundaries; actual
  occupied states are separately reported. Discretization depends on the
  included rows. Learn edges on training data only if using this in a
  held-out predictive study; the current API does **not** implement a
  train/test discretizer fit/reuse contract.

### Missingness and sample support

- `complete_case` (default): discard a row if *any requested variable* is
  `missing`, `explicit_null`, or `unknown`. Count exclusions.
- `as_category`: retain those generic states as separate discrete categories;
  `source_reason` stays diagnostic metadata, never a second implicit feature.
- `error`: raise on any missing state.
- The `empty` state is a distinct categorical level, never silently null.
- `multiple` states are rejected until a caller explicitly aligns or
  aggregates the occurrences; they are not new independent samples.
- `min_samples` defaults to 2, but does not claim adequate power.
- Conditional measures require at least `min_stratum_samples` (default 2)
  observations in **every observed joint conditioning stratum**. Otherwise
  `value=None` and `status='insufficient_conditioning_support'`. The user can
  explicitly relax this constraint but should not interpret singleton strata
  as evidence of conditional independence.
- `support` reports aligned/included/excluded rows, cell states, provenance
  reasons, conditioning strata, and distinct states. Sparse joint states emit
  an additional warning. Missingness may be informative, so complete-case and
  `as_category` estimates describe different distributions.

### Result semantics

- `value` is a finite nonnegative number in units set by `log_base` (base 2:
  bits). `log_base` must be > 1. Absence of statistical support is represented
  by `None`, **never** fabricated zero.
- `object_id`, `observation_unit`, requested variables, discretization policy,
  estimator, missingness policy, and sample support accompany every result.
- Full, partial, and flattened source graph transforms preserve the original
  paired-observation sample counts unless a separate, explicit observation
  transformation is requested. This implementation does not evaluate a
  topology's causal implications.
- Reports are deterministic, detached JSON-compatible dictionaries. No raw
  observation values, external calls, database writes, or source mutations.

## Test and acceptance scope

`tests/test_ml_uw_32_information_uncertainty.py` covers the XOR counterexample
(`I(X;Y)=0` and `I(X;Y|Z)=1`), exact binary entropy, deterministic dependence,
multiple conditioning variables, category type separation, empty/missing/null
provenance, missing policies, quantile ties, equal-width bins, conditioning
stratum support, malformed requests, and preservation under flattening and
non-tree graph topologies.

**Not implemented:** continuous-density/differential entropy estimators, KNN,
bootstrap intervals or permutation significance tests, automatic node-to-
observation alignment, inference of dependence edges from MI, learned graphical
models, UI controls, or new SQLite schema. These belong to separately reviewed
future sprints and should not be implied by the present API.
