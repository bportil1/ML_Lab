# HIA-08 — Empirical information conditioned on preceding structural levels

## Responsibility and ownership

ML_Lab owns scientific estimation and exact SU-05 observation bindings. The
existing PAH_Shared engine owns graph rendering, viewport and selection. HIA-08
adds an opt-in inspector and a source-pinned read-only endpoint, without changing
source topology, existing node/branch/level inspectors, or any persisted data.

Entry point: `analyze_hierarchical_conditional_level(...)`; schema
`ml-lab.hia-conditional-level-information@1`; route
`POST /workbench/structured/hierarchy/conditional-information` with the
explicit `X-ML-Lab-HIA: explicit-conditional-level-information` header.

## Exact structural scope

HIA-02 defines the structural level by rooted depth (tree/forest) or DAG
longest-path depth. `level_node_ids` and `preceding_level_node_ids` retain **all
original source IDs**. Each selected-level and previous-level variable must
be associated with a specific observed column by the operator. The entire
original level is not treated as observed when some nodes are unmapped.

- `mappings`: unique `{node_id, variable_id}` pairs **at level k**.
- `preceding_mappings`: unique pairs **strictly below level k**. For k>0,
  at least one must be supplied. For k=0 it must be empty.
- `target` (optional): unique mapped source node outside level k, not repeated
  in the current or conditioned subset. A target from an earlier structural
  depth is allowed when it was **not** used as a conditioner. This allows
  measuring how deeper levels predict an explicitly designated output while
  avoiding direct target leakage into conditioning variables.
- No variable/node may appear twice in the combined mapping. For floating
  observations the operator must configure explicit numeric discretization.
  Repeated-value aggregation policies remain explicit.

## Estimands (bits)

Let **X** represent the *joint vector of mapped level-k variables*; **Z** the
*joint vector of mapped preceding-level variables*. This is not the full
structural level unless all its source nodes were observed and mapped.

- `level_conditional_entropy_bits`: `H(X | Z) = H(X,Z)-H(Z)`
- `level_prior_mutual_information_bits`: `I(X;Z)=H(X)+H(Z)-H(X,Z)`
- `conditional_total_correlation_bits`: `sum_i H(X_i | Z) - H(X | Z)`
- Node-level `H(X_i | Z)` and `I(X_i; X_{-i} | Z)` for all mapped nodes
- Optional `target_incremental_information_bits`:
  `I(Y;X | Z)=H(X,Z)+H(Y,Z)-H(Z)-H(X,Y,Z)`.

These are direct empirical discrete plug-in estimates, with **no conditional
independence assumption**. HIA-09 will introduce explicitly selected
factorization models as separate labeled estimates. None of these measured
associations establish causality or information flow along source edges.

## Cohorts and availability

The base estimate uses a complete-case cohort covering **X and Z**. If Y is
requested, it uses a second independent complete-case cohort covering
**X, Z and Y**. Each set of entropy terms within an estimand uses the **same
rows**; the two results are not to be subtracted unless those cohorts happen
to match. Each report includes support, excluded-row reasons, joint-prior
strata and under-supported-strata counts. A joint-prior stratum with fewer
than the explicitly required `min_stratum_samples` blocks estimation and
returns null estimates and `insufficient_conditioning_support` status.

Sparse/high-dimensional categorical joint states produce an explicit warning;
plug-in estimates can suffer severe finite-sample bias. HIA-08 does not
impute unavailable estimates, assume independent prior nodes, invent missing
observations, sample the original graph, or silently discard source identities.

## User workflow

1. Open a verified source under **ML_Lab → Data → Structured Uncertainty**.
2. Click **Explore source graph** and **Inspect level**.
3. In **Conditional level information**, add selected-level node/column
   mappings. Add explicit earlier-level mappings with the `Earlier level Z`
   role. Optionally map an output with the `Target Y` role.
4. Configure repeated-value handling, numeric discretization, minimum
   complete rows and minimum support per joint-prior stratum.
5. Click **Compute conditional information**. Review both cohort supports,
   node-conditioned statistics and JSON provenance; optionally export JSON.

Existing graph presentation, node/neighbor/branch/whole-level statistics,
exports, navigation and source identities are not replaced.

## Scientific acceptance

Synthetic tests include balanced independent predictors, target information,
confounding eliminated by conditioning, joint-XOR conditioning, sparse joint
prior strata, differing optional-target support, float binning, repeated
values, source-only input, DAG depth and an inventory of over 5,000 source
level nodes with only two explicitly mapped variables. The Flask route
requires a live selection cookie, explicit header and exact source revision.
Browser acceptance and actual installed PAH integration must be checked on
the user's machine; isolated unit tests do not substitute for live UI checks.
