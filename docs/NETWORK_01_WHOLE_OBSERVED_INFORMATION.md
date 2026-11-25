# NETWORK-01 — Whole-network empirical information workflow

**Owner:** ML_Lab scientific analysis and UI; **renderer:** the installed **single PAH_Shared SU-02 graph engine**. No PAH host / pyPIQUE / HSQA_DBN changes. Additive after SU-11 and ML_Lab BUG-01 single-file intake.

## Why the earlier workflow looked incomplete

The existing **Feature Profiling → Option B** form computed a chosen `H(X)`, `I(X;Y)` or `I(X;Y|Z)` only. The **Data → Original structure** shared graph showed source topology, while SU-10 attached evidence to one *selected original source node* using explicit SU-05 mappings. These were not an all-observed-variable relationship-network workflow. A source edge cannot be treated as a statistical observation or a learned dependency.

## User journey

1. In **Structured Uncertainty → Data**, choose **Browse project files** and open an existing local `.csv`, `.tsv`, `.json` or `.xml` source using the BUG-01 intake. The source is read-only, project-scoped, and SHA-pinned.
2. For the **original hierarchy**, choose **Explore source graph** under Data (requires the locally installed PAH_Shared SU-02 renderer). This works for source-only XML, JSON and AST graphs even with **zero** aligned observations.
3. To evaluate a **whole empirical observed-variable network**, follow **Feature Profiling → Whole-network analysis**. This is an entirely separate statistical graph, not the original source graph. It requires *at least two aligned observed columns and sample records*. A raw standalone XML tree does not establish them. First collect/align XML records through the existing XML collection tools if empirical measurements are intended.
4. Leave **Variable IDs** blank to include *every* eligible observed column (up to the explicit cap), or enter original column names to bound the calculation. Optional global conditioning columns `Z` are excluded from graph endpoints. No field names or graph edges are automatically mapped to one another.
5. Optionally configure typed numeric bin definitions, missing-value treatment, per-pair minimum support, and the graphical display threshold. Press **Compute entire observed-variable network**. Nothing computes on navigation.
6. Inspect the full, symmetric all-pairs MI matrix (or CMI matrix if Z is set), support and status for every pair; select nodes or edges in the shared interactive network to inspect observed variable IDs, precise estimates, dropped record counts, cardinalities and estimator warnings.
7. Change **Display threshold** to show/hide observed associations without rerunning analysis or deleting the full numerical matrix. **Export complete report JSON** preserves every attempted pair and full provenance. Reports are held in the browser until explicit export: no database writes, structure changes, source mutations, cross-project reads or background tasks.

## Scientific contract

- Core: `ml_lab.structured.network_information.analyze_observed_network(source, …)`.
- Schema: `ml-lab.observed-information-network@1`. Stable synthetic *display* node IDs have a one-to-one report mapping to original exact variable IDs; no source-node identity is implied.
- **Every unordered observed-variable pair** is computed, using the existing ML-UW-32 empirical-discrete plug-in CMI implementation. With empty Z this is MI. Non-empty Z is a *single explicitly declared* conditioning set shared by every pair. This is **not** CMI given all other variables by default.
- Observations, never graph nodes, are the samples. Under default pairwise complete-case handling, *sample support may differ across matrix entries* and is disclosed per pair. Missing states can alternatively be represented as separate categories, or rejected.
- Explicit floating-point binning is required. Numeric-looking **CSV/TSV strings are not silently converted** into continuous numbers; they remain exact categorical strings unless a separate explicit typed preprocessing workflow is used. Each bin definition is fitted only once over the available source observations. Those cutpoints are descriptive; never reuse them in predictive holdout evaluation without fitting on training rows only.
- The diagonal displays single-variable plug-in entropy `H(X)` and support. Off-diagonal is `I(X;Y)` or `I(X;Y|Z)` in bits. Values of 0.0 are measured zeros, not unavailable states. No significance testing, unbiased estimates, P-values, confidence intervals, causation, directionality, propagation or fitted Bayesian-network structure is asserted.
- A *separate full-joint exploratory statistic* for **up to 12 variables** calculates common-sample joint entropy and **total correlation** `Σ H(Xᵢ) − H(X₁,…,Xₖ)`, with explicit joint-state/support bounds. This can reveal aggregate dependence (including a three-way XOR even with all marginal pair MI=0), but **does not localize the higher-order interaction** or justify source-graph pruning. For larger joint dimensions it is explicitly unavailable. Joint total correlation is unconditional, even if pairwise CMI is requested.
- Each edge is an **undirected empirical statistical association** with support, not an original structural, causal or explicit probabilistic edge. `graph_role = empirical_statistical_association` is distinct from SU-04's `source_structure`. PAH_Shared owns all layout, rendering, pan/zoom, overview, keyboard selection and graph interaction. No copied renderer code.
- Dense graphs can use PAH_Shared's visibly labeled overview/collapse behavior. The complete matrix retains **all** attempted pair results regardless of visualization filtering/collapse.

## Explicit interactive limits

- 2–128 selected endpoint variables; at most 8 declared conditioning variables.
- Up to 10,000 aligned observation rows.
- `number_of_pairs × number_of_rows ≤ 10,000,000`. An over-limit request **fails explicitly**; the system never silently samples, selects the strongest links before estimating, or drops variables.
- 5,000 distinct joint states for each plug-in pair, 5,000 for the optional <=12-variable joint summary.
- Every pair reports `status`, support, excluded rows, conditioning strata, cardinality and sparse-state warnings. Insufficient samples or stratum support has a null estimate, not a fabricated zero.
- Interactive POST is no larger than 32 KiB and requires the explicit same-origin header plus the currently selected exact source SHA. Source bytes are verified anew per run. Renderer request is the existing PAH_Shared asset served by the local ML_Lab package; no CDN.

## Validation

Tests: `tests/test_structured_whole_network.py`. Synthetic all-pairs matrix, independent marginals, XOR CMI, three-way XOR total correlation, finite sample/missingness statuses, explicit floating bins, repeated-value rejection, read-only repeatability, 80-column exhaustive estimation, complexity guard, source/association graph separation and selected-source HTTP route (when Flask is installed). JavaScript syntax and PAH_Shared SU-02 view contracts have separate local checks.

**Remaining live gates:** Install current ML_Lab source in PAH's actual Python environment, verify PAH_Shared SU-02 is installed, run project-local XML/CSV examples, confirm browser pan/zoom/overview/matrix/selection, long-label behavior, full-suite and upstream PAH browser acceptance. This patch does not claim those gates passed in this container.
