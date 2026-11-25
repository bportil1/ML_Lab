# HIA-06 — Exact structural branches with observed-subset information

## Scope and ownership

The **ML_Lab** standalone repository owns this opt-in numerical analysis. The
existing **PAH_Shared** SVG graph engine still owns rendering, drag/zoom,
selection, scaling, overview and export behavior. No sources, SQLite files,
experiments, or project outputs are written by this feature. This patch does
not modify PAH's `modules/` folder or any other repository.

### Input and UI path

1. Data → Structured Uncertainty → **Explore source graph**.
2. Select an original source node and choose **Inspect branch** in the HIA-03
   hierarchy navigator. Branch selection is available for verified rooted
   **trees and forests**, not for multi-parent DAGs or cyclic source graphs.
3. The **Branch information** panel receives the entire exact source branch
   membership from HIA-02. Select an original branch node (including by
   clicking a member on any HIA-03 member page), and **explicitly** associate
   that node with one original aligned observation column; add more mappings.
4. Optionally provide a separate original node and observation column outside
   the branch (often its structural parent), explicit numeric discretizations,
   occurrence policy, and minimum support.
5. Press **Compute mapped branch information**. There are **no automatic
   estimations** on hover, graph open, or selection. Download the complete
   JSON report with original IDs, observation binding plan and fingerprints.

The report retains `branch_node_ids` and `branch_node_count` for the **entire
source branch**. `mapped_node_count`, `unmapped_node_count` and
`statistical_scope` identify the **subset with explicitly bound observed
columns**. Statistics are never implicitly attributed to unmapped branch nodes.

## Estimands and support

Let \(B=(X_1,\ldots,X_k)\) be the explicitly mapped *observed-variable*
subset of the chosen structural branch, not the source nodes themselves.
Complete-case rows are aligned SU-05 **observation units**.

* `joint_entropy_bits`: \(H(B)\) (empirical categorical plug-in).
* `sum_atomic_entropy_bits`: \(\sum_i H(X_i)\) on the **same complete-case cohort**.
* `total_correlation_bits`: \(TC(B)=\sum_i H(X_i)-H(B)\), a nonnegative
  aggregate of within-subset dependence. It is **not** the sum of pairwise MI,
  and is not proof of causality or any graph-direction assertion.
* `per_mapped_node`: marginal \(H(X_i)\), unique conditional remainder
  \(H(X_i\mid B\setminus X_i)\) and MI with the other mapped branch variables
  \(I(X_i;B\setminus X_i)\). For a single mapped variable, the latter is zero.

When a **separate, explicitly mapped context** \(Z\) is requested:

* `branch_conditional_entropy_bits`: \(H(B\mid Z)\).
* `branch_context_mutual_information_bits`: \(I(B;Z)\).
* `conditional_total_correlation_bits`:
  \(\sum_i H(X_i\mid Z)-H(B\mid Z)\).

`context_is_direct_source_parent` describes source topology only; the context
could be a parent, a sibling, or another original node **outside the branch**.
It is **not** an assumed probabilistic parent. These conditional quantities
are empirical joint estimates, not a conditional-independence factorization
and not an incremental predictive contribution about a target. The latter is
planned for HIA-08/09.

The report includes distinct-state counts and flags joint-state sparsity
relative to complete-case support, without silently truncating source nodes.

Every metric uses an explicit, common complete-case cohort for its own joint
estimand. Branch-only and context-conditioned cohorts may have **different
support**; both are labelled separately. Finite-sample plugin entropy is biased,
and high-dimensional state spaces may be sparse: users must not interpret a
finite value as statistical significance. Observation rows are not automatically
independent simply because the source has many nodes.

### Missingness, repeats and numeric observations

* Incomplete rows are reported as exclusions, not zero-filled.
* Unresolved repeated values result in explicit `unresolved_repeated_values`
  rather than a partial entropy estimate. An explicitly requested numeric mean
  policy uses the existing SU-05 binding implementation and is recorded.
* Floating-point values require explicit quantile/equal-width binning, with
  the fitted definitions preserved in the report. Booleans, integers and strings
  are discrete categorical states; source edges are never converted to samples.
* No underlying statistical estimator or source graph is modified. This adds
  a separate versioned report: `ml-lab.hia-branch-statistics@1`.

## Source contracts and verification

- `src/ml_lab/structured/hierarchical_branch_statistics.py`: scientific entry
  point `analyze_hierarchical_branch`.
- `src/ml_lab/ui/workbench_structured_routes.py`: source-fingerprint-verified,
  explicit JSON endpoint `/workbench/structured/hierarchy/branch-information`.
- `src/ml_lab/ui/static/ml_lab_hia_branch.js`: new client inspector; HIA-03
  emits a semantic scope event without altering the shared viewer.
- `tests/test_hia06_branch_statistics.py`: independent/dependent, context,
  missingness, repeated cells, partial mapping, invalid topology, source
  fingerprint, no-persistence, and client contract cases.

Validation from the standalone ML_Lab checkout:

```bash
PYTHONPATH=src python -m pytest -q tests/test_hia06_branch_statistics.py
node --check src/ml_lab/ui/static/ml_lab_hia_branch.js
```

Existing HIA-04 atomic and HIA-05 neighbor statistics, HIA-01/02 contracts,
HIA-03 graph controls, and SU-01–SU-11 remain available.

Next: **HIA-07**, reproducible full-level joint information (with separate
level membership and statistical scope), followed by HIA-08 conditional-level
information. HIA-06 does not silently perform those later analyses.
