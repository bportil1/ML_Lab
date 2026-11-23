# ML-UW-33 — Probabilistic model specification (Option C1)

**Baseline:** ML-UW-32. **Status:** Explicit, validated model *contract* only. Fitting and inference remain ML-UW-34.

## What is represented

`ml_lab.structured.probabilistic.ProbabilisticModelSpec` is a new, versioned, immutable, JSON-compatible **discrete Bayesian-network specification**, separate from a `StructuredObject`'s source topology and from any statistical association graph. No XML, AST or pyPIQUE semantics are imported. A model can be an explicitly authored forest, rooted tree or DAG regardless of the source object's topology. Cyclic model arcs are always rejected, even if the source is a cyclic network.

`MODEL_SCHEMA = "ml-lab.probabilistic-model@1"`. All models require a *supplied, validated* `StructuredObject` at creation and reload. The specification saves both `source.object_id` and a SHA-256 fingerprint of exact source content; the latter protects against an externally supplied, stale or reused object ID. The source itself is never rewritten or automatically modeled.

For v1, every model variable must have a distinct `observed_variable` in the source's **aligned observation rows**. The names are explicit; model variable IDs and source node IDs need not coincide. `source_node_ids` are optional traceability references to actual source nodes. Model arcs must be authored explicitly; their optional `source_edge_ids` are individually checked for valid endpoints and direction against the referenced model-variable mappings. *An AST/XML parent edge does not automatically create a probabilistic edge.*

## Example specification (not fitted by ML-UW-33)

```python
from ml_lab.structured import ProbabilisticModelSpec

# `source` is a validated StructuredObject with aligned string-valued
# observation columns `x` and `y`; caller explicitly defines this graph.
model = ProbabilisticModelSpec.from_dict({
    'schema': 'ml-lab.probabilistic-model@1',
    'model_type': 'discrete_bayesian_network',
    'variables': [
        {'id': 'X', 'observed_variable': 'x', 'states': ['0', '1']},
        {'id': 'Y', 'observed_variable': 'y', 'states': ['0', '1']},
    ],
    'arcs': [{'parent': 'X', 'child': 'Y'}],
    'conditionals': [
        {'variable_id': 'X', 'family': 'categorical', 'status': 'unfitted', 'rows': []},
        {'variable_id': 'Y', 'family': 'categorical', 'status': 'unfitted', 'rows': []},
    ],
    'missing_policy': 'complete_case',
}, source=source)

print(model.audit)  # support counts, topological order, explicit assumptions
serialized = model.to_json()
restored = ProbabilisticModelSpec.from_json(serialized, source=source)
```

A specified conditional uses `status: "specified"` with **all** parent assignments, and each row contains `given` (parent-state mapping) and `probabilities` (child-state mapping). Root variables have a single row with `given: {}`. Probabilities must be finite and in `[0, 1]`, with each row summing to one within `1e-9` absolute tolerance. Boolean values are not probabilities. `unfitted` conditionals must contain no rows, so future estimation never confuses unestimated parameters with a zero or uniform distribution. Mixed specified/unfitted models are labeled `partial`. Input row order does not change model identity.

## Scientific assumptions and exclusions

- **Only finite categorical variables** are supported in v1, with 2–64 explicitly labeled states per variable and at most 64 model variables. Numerical values require an explicit, documented discretization before a probabilistic model can use them. Empty-string observations require the empty label in the declared domain.
- **Only tabular conditional probability distributions** are supported in v1. Gaussian, logistic, neural, undirected factors, latent variables, mixed continuous/discrete, and nonparametric likelihoods are rejected rather than silently converted.
- **CPT bounds:** each conditional can require at most 4096 parent-state assignments. This guards specification expansion; it is not an inference complexity guarantee.
- **Missing-data policy:** explicit `complete_case` or `reject`. Missing, explicit-null, unknown and repeated cells are either excluded from complete-case support or rejected. No zero filling, null-category coercion, imputation, missingness mechanism assumptions or inferred marginal distribution. A model can be specified with zero complete rows; it cannot be meaningfully fitted from them without future estimator support checks.
- **Observation support:** model audit contains total rows, complete rows, excluded rows, and counts per variable. It does *not* establish i.i.d. sampling, statistical independence, causal interpretation or good parameter estimates. The joint factorization follows the *authored* model arcs, not observed correlations or original source edges.
- **No training or inference:** a complete table may be marked `specified`, but **no `fit`, `predict`, `query`, posterior, likelihood, posterior uncertainty or inference API** is available in ML-UW-33. ML-UW-34 must implement and test any such operation separately.
- **No implicit integration:** no UI routes, global registries, local SQLite changes, cross-module database reads, automatic processing or runtime files. Source modules remain data owners.

## Strict validation and provenance

- Reject unknown fields, duplicates, self-loops, cycles, foreign endpoint/node/edge references, duplicated observation mapping, unaligned samples, out-of-domain values, unsupported conditional families and incomplete/invalid probability tables.
- The model's `model_id` is a content-derived SHA-256 identifier including source identity, specification, missing-data policy and author provenance. Explicit IDs must match the computed content hash.
- Model JSON rejects duplicate keys and nonfinite values. Reimport requires the original verified `StructuredObject`; replacing its contents while preserving its caller-supplied object ID is detected by the source fingerprint.
- Immutable internal state; `to_dict()` and `audit` return detached copies. No read/write side effects, no dependency changes.

## ML-UW-34 entry gate

1. Choose a bounded subset of categorical BN structures for fitting and inference, with explicit sample-support limits and smoothing choices.
2. Do **not** assume observed rows are i.i.d. or missing at random. Record actual missingness and effective support in any fit.
3. Estimate unfitted tables only through an explicit user action with reproducible configuration and project-local provenance.
4. Implement exact inference only where bounded and verified by known-distribution fixtures. Never use a source graph edge as a conditional without explicit model specification.
5. Keep model specifications distinct from fitted parameter artifacts and query results; document any new model schema version if semantics change.

## Verification / rollback

`pytest -q tests/test_ml_uw_33_probabilistic_model_spec.py` plus existing ML-UW structured regressions. Revert this additive code, tests and internal note without touching datasets, database schema or existing XML workflows. Headless results are not PAH/Flask/browser acceptance.
