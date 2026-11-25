# HIA-10 — Matched-cohort whole-network information accounting

## Ownership and scope

The standalone **ML_Lab** repository owns this optional scientific analysis. PAH_Shared retains ownership of graph rendering, camera operations, and source-node visualization. HIA-10 does not re-parent nodes, alter edge identities, infer causal/statistical edges from topology, modify source data, or write databases. The existing HIA-01–09 estimators and inspectors are unchanged.

## Scientific contract

Entry point: `ml_lab.structured.analyze_hierarchical_network_accounting`.
Schema: `ml-lab.hia-network-information-accounting@1`.
HTTP endpoint: `POST /workbench/structured/hierarchy/network-accounting`, with header `X-ML-Lab-HIA: explicit-network-information-accounting`, source-revision pin, explicit node-to-column mappings, optional external target mapping, occurrence policy, numeric bins and minimum rows.

HIA-02 supplies **structural depth** for each original node: rooted depth in trees/forests, deterministic longest-path depth in DAGs. HIA-05/SU-05 supplies *aligned observed values*, but **no statistical measurement is invented** for source-only or unmapped nodes. The complete structural inventory is returned even when statistical coverage is partial. A target is another explicitly mapped observed source node, distinct from all level-variable mappings. Empty structural levels or levels without observed mappings do not receive invented estimates.

### Matched-cohort policy

All reported entropies, target increments and cross-level MIs use **exactly one complete-case cohort** containing every mapped level variable and optional target. This avoids the common error of subtracting estimates from different observations. With missing rows, repeated values, insufficient support, or numeric values without explicit discretization, the status makes that limitation explicit. The in-sample plug-in estimates are descriptive and can be biased on sparse/high-dimensional joint state spaces.

Let `X_k` denote the joint vector of mapped observed variables at structural depth `k`; `X_{<k}` denotes *mapped observed variables* from strictly earlier depths. Then:

- Level uncertainty: `H(X_k)`.
- Accumulated observed uncertainty: `H(X_≤k)`.
- Incremental joint uncertainty: `H(X_k | X_<k) = H(X_≤k) − H(X_<k)`.
- Overlap with prior levels: `I(X_k; X_<k) = H(X_k) − H(X_k | X_<k)`.
- Within-level total correlation: `Σ_i H(X_ki) − H(X_k)`.
- Whole-mapped-network total correlation: `Σ_i H(X_i) − H(X_all)`.

If external target `Y` is mapped, the same cohort also produces:

- Cumulative target information: `I(Y; X_≤k)`.
- **Incremental target information:** `I(Y; X_k | X_<k) = I(Y; X_≤k) − I(Y; X_<k)`.

Cross-level diagnostics report `I(X_i; X_j)` for each pair of **mapped** structural levels. They are overlapping descriptive quantities and **are not added** to derive cumulative or total information. No probabilistic or conditional-independence factorization is assumed: the factorized comparison remains HIA-09.

## Operation

Open **Data → Structured Uncertainty → Explore source graph → HIA-10 Whole-network information accounting**. The graph stays read-only. Choose an original source node, map an aligned observed column, and repeat across levels. Optionally map a separate target. Explicitly select binning and repeated-value handling before pressing **Compute whole-network information**. Results include level increments, cross-level pairwise MI, support, complete structural inventory, unmapped counts, source/binding fingerprints, and JSON export. No analysis is performed automatically.

The form intentionally does not infer variable mappings from node labels or topology. For many variables, scientific estimates may be computationally expensive and under-supported; no inaccurate fallback to sample-based or additive estimates is performed. HIA-11 will add graphical overlays and comparisons, and HIA-12 will include deployment-scale browser validation.

## Regression acceptance

`PYTHONPATH=src python -m pytest -q tests/test_hia10_network_accounting.py`

Regression fixtures check independent levels, perfect redundancy, zero incremental target information beyond a prior predictor, missing target rows, incomplete support, duplicate mappings, original source preservation and JSON serializability.
