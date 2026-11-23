# ML-UW-36 — Structured uncertainty workbench UI

Depends on ML-UW-27–35. This release exposes their **existing** abstract-object
APIs within the five current ML_Lab sections; it introduces no sixth primary
tab, schema migration, cross-module database access, new learning algorithm, or
automatic analysis on page navigation.

## Stage mapping

- **Data:** select a project-relative `.json` artifact via the validated
  `StructuredObject`, generic JSON, typed graph, or canonical XML structure
  adapter; inspect source topology. Raw `.xml` is processed in the existing XML
  workflow first. Generic nested JSON creates an instance tree but **not**
  independent sample rows.
- **Feature Profiling:** Option A structural and aligned-observation statistics,
  including nested variance; Option B discrete entropy, conditional entropy,
  MI, and CMI with explicit conditioning, missingness and numeric binning.
- **Representation:** full, depth, aggregate, subtree, and flat transformations;
  optional observation-variable projection is separate from topology changes.
- **Comparison:** matched source population, shared group-blocked holdout,
  training-only information estimates, structural retention and categorical
  holdout scores from ML-UW-35. Ungrouped splitting requires acknowledgement.
- **General ML Tools:** explicit finite-categorical Bayesian-network model
  validation, independent *unfitted* draft, empirical CPT fitting, bounded
  exact queries. An independent draft does not infer graph arcs.

## Scientific and execution guarantees

Source graph links, empirically estimated dependencies and probabilistic model
arcs are **distinct artifacts**. Source nodes are not sample rows; missingness
is not imputed silently. Comparisons distinguish structural retention from
observation-feature retention, without attributing predictive improvements to
flattening alone. Estimator status and sample support are displayed rather than
silently inventing an estimate.

- Local JSON sources only, maximum **2 MiB**, at most **10,000** observation rows
  in this synchronous UI; scientific core imposes additional limits.
- Rejects symlink traversal, paths outside the configured project root, invalid
  JSON and stale source fingerprints. Only path, adapter and SHA-256 are stored
  in a project/mount-specific HttpOnly SameSite cookie; the cookie is **not**
  an access-control boundary. Source content remains local.
- No new SQLite tables, automatic training, file writes, network requests or
  cross-repository source inspection. Reports and transformed representations
  can be explicitly exported as JSON. The model JSON is also shown in the
  editor after draft generation or fitting.
- UI navigation does not call statistical engines. Only explicit POST actions
  run analyses. An unavailable or unsupported operation raises a visible error.
- Existing workbench pages, five primary navigation links and prior XML tools
  are retained. New contextual links open the structured subworkflow.

## Validation and remaining acceptance

Headless tests validate path confinement, SHA source pinning, absence of implicit
sample construction, Option A/B/C calls, full/flat/hierarchy transformations,
matched comparisons, and template compilation. Flask browser checks are
included but skipped on systems without optional `ml-lab[ui]` dependencies.
Live mounted PAH acceptance is still necessary.

ML-UW-37 will pilot exported AST and pyPIQUE network artifacts via existing
module APIs/adapter contracts. ML-UW-38 covers larger realistic workloads,
performance bounds, browser behavior, and end-to-end integration. Structure-
aware neural models and automatic learned graph topology remain deferred.
