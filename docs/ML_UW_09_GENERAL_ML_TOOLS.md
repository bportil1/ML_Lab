# ML-UW-09 — General ML Tools consolidation (internal)

## Scope

UI consolidation only. The existing experiment editor, RBM task, GAN task,
optimization POST runner, and synthetic dataset runner remain authoritative.
The unified five-tab navigation retains Data → Feature Profiling →
Representation → Comparison → General ML Tools. No model engine, source dataset,
trained weights, or experiment revision is migrated or rewritten.

## Launcher behavior

- Four grouped panels: Models, EBM, Optimization, Generative.
- The current project-owned ExperimentStore supplies at most 200 recent
  experiment revisions (latest only). Each group displays at most eight links.
- Model selection links explicitly preserve experiment ID, model family,
  method, and revision context without starting a run. Revision validation
  and current dataset checks remain with the existing model editor.
- Only configured methods with numeric control descriptors **and** allowed
  objective metrics appear under Optimization. RBM and GAN do not currently
  advertise optimization objectives, even though the general optimization
  infrastructure exists.
- Generative workflow opens the existing statistical/conditional GAN/RBM
  synthetic runner. Earlier weights are not reused for these generation jobs.
- Existing downstream result and artifact views remain unchanged.
- Opening this page never verifies source data bytes, dispatches a job,
  changes a model spec, or selects a dataset for the user.

## Safety, isolation, and rollback

Only the project-local ExperimentStore is consulted; the launcher does not
traverse PAH or other module databases. The catalog is bounded and read-only.
The addition requires no schema migration, new dependency, or global state.
Links are built using Flask's mounted Blueprint so PAH URL prefixes work.
Removing this patch leaves experiment revisions, runs, and artifacts intact.

## Known limitations

The recent catalog is intentionally limited to 200 experiments and eight
visible entries per tool. Older records remain available through existing
experiment editors. Flask-dependent HTTP tests run only where the optional
UI dependency is installed. Optimizable means that the configuration exposes
supported controls/objectives; source-dataset freshness still must be
verified by the existing route before a job is allowed.
