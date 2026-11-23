# ML-UW-34 — Categorical probabilistic estimation and exact inference (Option C2)

**Baseline:** ML-UW-33. **Status:** Headless, explicit finite-categorical Bayesian-network fitting and bounded exact inference. **No UI, database migrations, source parsing or background computation.**

## Separation of scientific objects

- `StructuredObject` remains an immutable, validated *source* graph plus aligned observation rows. Its XML/AST/JSON/pyPIQUE origin is not inspected by the fitting or inference engine.
- `ProbabilisticModelSpec` (ML-UW-33) describes an **explicit** categorical DAG, observed-variable mappings and CPT status. No source edges are automatically converted to Bayesian-network arcs and no causal interpretation is implied.
- `FittingResult` retains a new, separately identified, fully specified `ProbabilisticModelSpec` **and** a distinct fit report; the original model and source are not modified.
- `query_exact()` returns a standalone JSON-compatible posterior/evidence report, not a revised model or training dataset. No global state, writes, cross-project database access or model discovery.

## Fitting API

```python
from ml_lab.structured import (
    StructuredObject, ProbabilisticModelSpec, fit_categorical_model,
    query_exact,
)

# source: validated StructuredObject with aligned categorical observations
# spec: explicitly authored, validated ProbabilisticModelSpec bound to source
fit = fit_categorical_model(
    spec, source=source, pseudocount=1.0,
    min_complete_rows=2, min_parent_rows=1,
)
fitted_spec = fit.model
fit_audit = fit.report
posterior = query_exact(
    fitted_spec, variables=['Y'], evidence={'X': '1'},
    max_assignments=65536,
)
```

Only *unfitted* CPTs are estimated by default; any specified CPTs are preserved. Trying to fit a completely specified model fails unless `replace_specified=True` is explicitly requested. An intentional refit produces a new model identity and carries its input model ID into provenance. No disk artifact is created automatically.

### Parameter estimator

For each child state `s` and **observed** parent assignment `u`, fit uses complete-case counts `N(s,u)`:

\[
\widehat{P}(X=s\mid \operatorname{Pa}(X)=u)
=\frac{N(s,u)+\alpha}{N(u)+\alpha\lvert\mathcal{X}\rvert}.
\]

`pseudocount=0` (default) is the unsmoothed empirical-frequency/MLE estimator. A positive `pseudocount` is explicit, symmetric per-state Dirichlet smoothing (the corresponding posterior predictive mean). This is **not** a confidence interval, proof of the selected graph, model selection, causal inference, or structure learning.

- `min_complete_rows >= 1` requires at least that many fully observed joint rows across **all model variables**. It does not claim their statistical independence.
- `min_parent_rows >= 1` (default) requires real observations for **every** parent assignment of each estimated CPT. Sparse configurations cause an explicit error, rather than invented frequencies.
- `min_parent_rows=0` is permitted **only** with positive smoothing and explicitly allows prior-only CPT rows. Every such row is marked in the fit report and included in its count.
- Missing/unknown, `explicit_null`, and `multiple` cells follow the model's declared missing-data policy. Under `complete_case` a row is excluded globally; under `reject`, validation already refuses it. No imputation, collapsed empty-null category, MAR/MCAR assumption or EM algorithm.
- The report contains complete/excluded row counts, per-variable availability, parent-assignment counts, per-state counts, smoothing configuration, original/fitted model identities, source fingerprints, and in-sample complete-case log-likelihood (unavailable with explicitly counted zero-likelihood rows). Count support is **descriptive**; it does not prove i.i.d. sampling, representative samples, or calibrated uncertainty.
- Model CPT limits still come from ML-UW-33 (at most 64 variables, 64 categorical states/variable and 4096 parent assignments per CPT). Unfitted and partially specified models remain valid specifications, but queries refuse them until they are fully specified.

## Exact queries

`query_exact(model, variables=['Y', 'Z'], evidence={'X':'0'}, max_assignments=65536)` returns the **joint conditional** distribution of the requested variables, not separate marginal estimates. Every query variable and evidence variable must reference an explicitly declared, disjoint model variable with a valid categorical label. Evidence is conditioning, **not** a causal intervention.

Exact inference enumerates evidence-compatible joint states, multiplies the authored CPT factors and marginalizes all other variables. Probability accumulation uses log-sum-exp for numerical stability. The report includes:

- `status='ok'` and a normalized posterior list of `{given: {var: state}, probability: number}`, ordered deterministically by the query-variable and state order.
- The scalar evidence probability and `log_probability_evidence`; if the scalar underflows to `0.0` while the event still has positive probability, the log remains finite and `audit.probability_evidence_underflow` is `True`.
- `status='unavailable'`, `reason='zero_probability_evidence'`, `posterior=None` if the evidence is truly impossible. No zero or uniform posterior is fabricated.
- `enumerated_assignments`, `nonzero_assignments`, `max_assignments`, source/model reference, and a reminder that the query is not causal inference.

**Resource limit:** the maximum permitted enumeration budget is 131,072 assignments (default 65,536). Evidence-fixed variables are not multiplied into the enumeration size; unobserved variables are. Joint query outputs are capped at 4,096 state combinations. The engine rejects unsupported workloads *before* enumerating. There is **no approximate-inference fallback**, stochastic sampler or unreported truncation. Worst-case work is proportional to model-variable count times enumerated joint states; choose small graphs or strong evidence.

## Validation and scope limits

- Known two-node Bayes posterior, normalized joint probabilities, three-node collider, explicit impossible evidence, and rare-event log probabilities.
- Observed / missing / explicit-null / multiple-state rows; no ghost records or silent zero filling.
- Unseen parent configurations; explicit prior-only opt-in; `replace_specified` protection; invalid configuration types; stale/forged source fingerprints; complexity-budget rejection; immutable model and report; zero filesystem writes.
- No continuous likelihoods, latent/unobserved variables, learned dependencies, interventions, confidence/credible intervals, expectation-maximization, approximate inference, data partitioning, automatic scoring against held-out data or GUI yet. These are **not** silently substituted.
- Existing ML-UW-33 CPT validation and source-binding assumptions remain enforced. No schema migration for the source or probabilistic *specification*. New distinct report schema identifiers: `ml-lab.probabilistic-fit@1` and `ml-lab.probabilistic-query@1`.

## Reproducible verification

```sh
PYTHONPATH=src pytest -q tests/test_ml_uw_33_probabilistic_model_spec.py tests/test_ml_uw_34_probabilistic_fitting_inference.py
PYTHONPATH=src pytest -q tests/test_ml_uw*.py
```

ML-UW-35 will compare structural-preservation alternatives under matched observation units, support and evaluation budgets. ML-UW-36 will integrate scientific capabilities into the existing five-tab workbench. A passing headless test suite is **not** Flask/PAH browser acceptance.
