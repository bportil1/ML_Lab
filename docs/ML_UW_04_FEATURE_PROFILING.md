# ML-UW-04 — Active-dataset feature profiling and type-aware exploration

## Scope

This increment implements the **Feature Profiling** section on top of the
verified ML-UW-03 Data → workbench handoff. It uses A2's existing `profile_file`
for type inference, missingness, duplicate/quality diagnostics, distribution
summaries and pairwise Pearson, Spearman, MI and NMI. It does **not** replace the
profiling engine, the legacy Data Lab `/data/profile/*` routes or the model's
feature and target selection. There are no new dependencies or external calls.

- **New workbench pages:** `GET /workbench/feature-profiling` shows the active
  dataset or the reason no verified dataset is available. The explicit
  `?inspect=1` action profiles it on demand (not during unrelated navigation).
  The page offers per-column inferred type, current visualization role,
  missing rate, uniqueness, variance, zero rate and quality flags.
- **Distribution explorer:** a bounded histogram for continuous features or
  frequency bars for discrete features. Values are calculated on the same
  deterministic reservoir parameters as the A2 profile. Numeric variance and
  zero rate are added to the existing A2 numeric summary dictionary. Boolean
  zero/false rate is supplementary workbench metadata.
- **Pair explorer:** continuous–continuous scatter (max 240 deterministic
  plotted points), continuous–discrete conditional five-number-summary table,
  and discrete–discrete contingency table (max 8 displayed categories per
  axis). Unsupported identifier/excluded pairs are explained, not coerced.
  These plots describe associations; they do not establish causation.
- **Overrides:** `POST /workbench/feature-profiling/role` accepts `auto`,
  `continuous`, `discrete`, `identifier`, `exclude`. It validates both the
  dataset checksum and optimistic workbench generation; continuous is only
  allowed for an A2-inferred numeric column. `auto` deletes the override.
  These roles affect **workbench display only** and do not rewrite the
  dataset, historic DataProfile files, experiment revisions, or training inputs.
- **Machine-readable preview:** `GET /workbench/feature-profiling/json` requires
  a verified active dataset and returns the same bounded profile/report. This
  uses existing project-local data, not any network endpoint.

## Storage and privacy

Feature roles are stored in `workbench_feature_roles`, an optional, additive
`CREATE TABLE IF NOT EXISTS` extension in the **same per-project**
`analysis/ml_lab/experiments.sqlite3` database already owned by `ExperimentStore`.
The existing SQLite `PRAGMA user_version` remains **3**, so an older v3
binary can still open the project after rolling back this patch; it simply
ignores the extra table. Roles are scoped by opaque workbench instance, source
SHA-256, and column name. Source fingerprint/provenance revalidation is
mandatory: stale datasets cannot be profiled or edited, and the old override
is not applied to a newly selected dataset with a different checksum.

No source files are modified, no new profile bytes are saved, and no cross-
project or cross-module database is introduced. No `ui/blueprint.py` changes.

## Limitations and follow-up

- Statistical summaries use at most **5,000 deterministic sampled rows** for
  the new interactive view. A2 still inspects the whole source and can perform
  its existing comprehensive background profile job; this increment does not
  add a workbench cache, server pagination, or asynchronous execution.
- Visualizations are intentionally lightweight and do not introduce plotting
  libraries. Type-role overrides **do not recompute the A2 MI/NMI ranking**:
  the dependence table explicitly retains its source-inferred semantics.
- Browser behavior in a PAH host must be smoke-tested with the optional Flask
  UI installed; tests conditionally skip in a headless-only environment.
- A later ML-UW increment can add cached profile runs, a complete dependence
  matrix, larger-data pagination and transformation/model handoff of explicitly
  selected type rules. None of these is claimed implemented here.

## Acceptance checks

1. Confirm a project-local CSV/TSV (including exported XML collection data) is
   explicitly active and verified under Data.
2. Visit Feature Profiling, click **Inspect active dataset**, and inspect the
   per-column quality table, distributions and pairwise views.
3. Set a display role, reload, confirm the override persists; select `Auto`
   and verify that it is removed. No experiment revision or dataset changes.
4. Modify the source after selection; verify the workbench marks it stale and
   refuses both profiling and role updates. Reselect the source as a **new**
   dataset revision and verify older overrides are not silently inherited.
5. Confirm existing A2 profile routes and independent project databases remain
   usable. Verify the patch against an ML-UW-03 checkout before applying.
