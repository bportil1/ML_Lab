# HIA-09 — Explicit conditional-independence model comparison

## Role and source ownership

ML_Lab owns HIA-09 estimates; PAH_Shared continues to own the original source
viewer's graph interaction, layout and export. HIA-09 only adds an optional
inspector under HIA-08, a strictly opt-in local HTTP endpoint, and a separate
versioned scientific schema (`ml-lab.hia-conditional-independence-model@1`).
It does **not** edit the original source, change any graph node/edge ID,
create a database, fit a learned graph structure, or add a statistical edge.

Entry point: `analyze_hierarchical_factorized_level(...)`.
Local endpoint: `POST /workbench/structured/hierarchy/factorized-information`
with `X-ML-Lab-HIA: explicit-conditional-factorization`, an explicit
`assumption_confirmed: true`, and a SHA-verified source selection.

## Modeling contract

For an operator-selected level, let `X=(X_1,...,X_m)` be the **joint vector
of explicitly mapped observed columns** at that level and `Z` the **joint
vector of explicitly mapped earlier-level observed columns**. Source nodes
without observed columns remain present in the structural inventory, but are
not assigned invented measurements.

The existing HIA-08 empirical result is `P_emp(X|Z)`; its entropy estimates,
complete-case support, optional target cohort and warnings remain unchanged.
The HIA-09 alternative **assumes** the following factorization:

\[
Q(X\mid Z) = \prod_{i=1}^m P_{\mathrm{emp}}(X_i\mid Z_{\mathrm{parents}(i)}).
\]

All factors are empirical frequencies fit on the **same complete-case cohort**
`(X,Z)` as HIA-08. Source edges are never used to guess probabilistic parents;
the operator must explicitly acknowledge the assumption.

Two choices are available:

1. `shared_joint_prior`: each `Z_parents(i)` is the **entire mapped prior
   vector**. No assumption about independence among the prior variables is made.
2. `explicit_prior_parent_sets`: the operator supplies a mapping from **every
   mapped level-node ID** to an array of *distinct, exactly mapped earlier-level
   node IDs*. Empty arrays are allowed (unconditional factors). Unknown,
   duplicate, current-level, or source-only parent IDs are rejected.

Both modes assume `X_i` independence **conditional on their selected prior
parents under Q**. They do not assert that source topology proves it.

## Statistics (bits)

Using a single matched empirical cohort, HIA-09 returns:

- `H_P(X|Z)` — empirical HIA-08 conditional joint entropy.
- `H_Q(X|Z)` — the factorized model entropy averaged over observed `P_emp(Z)`.
- `E_P[-log₂ Q(X|Z)]` — the factorized **in-sample cross-entropy**, numerically
  equal to `H_Q(X|Z)` for these maximum-likelihood factor tables.
- `D_KL(P_emp(X|Z) || Q(X|Z))` — model-vs-empirical conditional KL divergence.
- `TC_P(X|Z)` — empirical within-level conditional total correlation.
- Per-level-node omitted-context term `H_P(X_i|Z_parents(i))-H_P(X_i|Z)`.

For in-sample maximum-likelihood factors, the following exact decomposition
holds to floating-point tolerance:

\[
D_{KL}(P_{emp}(X|Z) \| Q(X|Z)) =
TC_P(X|Z) + \sum_i I_P(X_i; Z\setminus Z_{parents(i)} \mid Z_{parents(i)}).
\]

Thus, with `shared_joint_prior` the model divergence is exactly the empirical
conditional total correlation. Under explicit parent subsets, the additional
gap quantifies **omitted prior information**, not just dependence among level
members. Per-stratum KLs and their support are included. Each stratum uses
its original encoded observed states; no graph identity is lost or sampled.

**Interpretation caveats:** these are *in-sample descriptive plug-in estimates*,
not independent/held-out model validation, an independence hypothesis test,
causal flow, or a learned probabilistic graphical model. Sparse high-dimensional
joint states can yield low or zero in-sample KL even with genuine population
dependence. Under-supported joint prior strata cause estimates to be reported
as `null` with the original HIA-08 status. Optional target information remains
separate and is never subtracted across different complete-case cohorts.

## UI workflow

1. Open a verified structured source under Data → Structured Uncertainty,
   select `Explore source graph` → `Inspect level`.
2. Configure the existing HIA-08 level and preceding mappings and its
   observation/discretization settings. HIA-08 still runs only on request.
3. Open **HIA-09 · Optional conditional-independence model comparison**.
4. Explicitly confirm the model assumption and choose full joint prior or
   supply an exact JSON map of level node IDs to prior-node ID arrays.
5. Click **Compare independence model against empirical joint**. The comparison
   is independent of the HIA-08 Compute button and does not modify it.
6. Inspect the model entropy/cross-entropy, empirical joint entropy, KL,
   within-level TC, omitted prior information, per-factor values, full stratum
   diagnostics and exported JSON. Every change to mappings, source, mode,
   assumptions or estimator settings invalidates the previous model result.

## Acceptance and rollout

Local scientific tests cover independent and redundant level variables,
correct prior parents, omitted parents, XOR joint-conditioning, optional
insufficient support, floating-point binning, model consent, malformed parent
sets, immutable source identity, JSON round-trip and preservation of HIA-08.
The UI requires a verified selection cookie, exact source SHA, POST header and
valid request field set. All prior HIA graph, atomic, neighbor, branch and level
interfaces remain intact. Browser interaction and PAH-hosted deployment require
live acceptance in the user's standalone local environment.

**Next:** HIA-10 will aggregate information across multiple levels into a
whole-network accounting view; HIA-09 does not derive global network entropy or
pretend all structural nodes have observations.
