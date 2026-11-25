# SU-08 — Controlled, Replayable Structural Transformation Candidates

**Status:** Additive headless implementation. **Owner:** ML_Lab. **Dependencies:**
ML-UW-27–30, SU-05–07. **No UI, shared renderer, filesystem, SQLite, external
service, model fitting, or automatic pruning changes.** The source is never
modified. SU-09 is responsible for empirically evaluating retained variance,
information, or predictive task utility; SU-08 does not report those estimates.

## Scientific distinction

A source graph, hierarchy or AST is **not** an empirical population. Removing
source nodes or graph edges does not, by itself, remove or change aligned
observation columns. All operations preserve the original observation IDs,
record count, order, sampling unit and *retained cell payloads*. No joins, row
multiplication, statistical binning, imputation, scoring, or value aggregation
occur. Candidate manifests explicitly distinguish **structural loss** (source
identities and edge relationships) from **feature impact** (SU-05 declaration of
node-to-observation relationships and any *explicitly requested* column drop).

## Versioned contracts

- `ml-lab.structural-candidate-plan@1`: original complete `StructuredObject`
  SHA-256, selected operation, exact selected node IDs or absolute depth,
  observation policy, optional pinned SU-05 binding-result SHA-256.
- `ml-lab.structural-candidate-manifest@1`: candidate ID, source/output
  object IDs and full fingerprints, serializable plan, before/after topology,
  exact removed node/edge IDs, structural collapse-to-boundary mappings
  and counts, input/output variables, affected
  SU-05 variable mappings, explicit drops, replay and interpretation limits.
- Deterministic, detached JSON representations support archival and review.
  JSON readers reject unexpected fields, duplicate keys and nonfinite values.
- `replay_structural_candidate` **recomputes from the original source and
  binding** and verifies the full manifest. A manifest alone is not a
  lossless inverse or a cryptographic signature.

## Allowed operations

| Operation | Selection | What changes | Scientific caveat |
| --- | --- | --- | --- |
| `preserve` | none | Identity: original object returned | No optimization claimed |
| `prune` | explicit disjoint branch roots | Remove entire selected rooted subtree(s) | Reject removal of all nodes; DAG/cycle pruning undefined |
| `aggregate` | absolute `max_depth` | Existing ML-UW-30 structural boundary count aggregation | **No** numeric/observed-value aggregation |
| `depth_reduce` | absolute `max_depth` | Existing ML-UW-30 depth-limited hierarchy | Observation values unchanged by default |
| `summarize` | explicit disjoint boundary nodes | Keep boundary node, remove descendants, attach per-kind and edge-count summary | Counts only; no statistical reconstruction |
| `flatten` | none | Existing ML-UW-30 flat structural view: retain nodes/attributes, discard edges | No graph-node-to-record join |

Only verified `tree`/`forest` source topologies can be pruned, summarized,
aggregated or depth-reduced. DAGs, cyclic networks, and undirected graphs can
be preserved or flattened but cannot be interpreted as having a unique parent
hierarchy. No display edge is treated as a learned/statistical/causal edge.

### Aligned observation policies

- **`preserve` (default):** all original observed columns remain, even where
  the referenced structural node was removed. The manifest reports this as
  at-risk *structural mapping*, not measured information loss.
- **`drop_exclusive` (opt-in):** requires a matching, validated SU-05
  `ObservationBindingPlan` and its exact derived binding digest. Drop a column
  **only** when every explicitly bound node ID for that column was removed.
  Preserve unbound columns, group/weight columns, and variables whose merged
  node binding spans both retained and removed nodes. Reject a candidate that
  would delete every observation column. This is a controlled feature-selection
  projection, **not** a claim that source-node pruning necessarily removes
  observed measurements or changes information content.

An ambiguous multiple-node binding is reported explicitly. SU-05's resolved
numeric means are *not* silently written over original repeated cells. No
observation variable is inferred by matching a label to a node name.

## Example, in an existing local Python environment

```python
from ml_lab.structured import (
    ObservationBindingPlan, VariableBinding, StructuredObject,
    StructuralCandidatePlan, source_sha256, binding_sha256, bind_observations,
    apply_structural_candidate, replay_structural_candidate,
    enumerate_structural_candidates,
)

source = StructuredObject.from_json(local_json_text)  # already read locally
binding = ObservationBindingPlan(
    source_sha256(source), source.observation_unit,
    (VariableBinding('measure_a', ('exact-source-node-id',)),),
)
digest = binding_sha256(bind_observations(source, binding))
plans = enumerate_structural_candidates(
    source, binding=binding, node_ids=('exact-source-node-id',), depths=(1, 2),
)
plan = StructuralCandidatePlan(
    source_sha256(source), 'prune', ('exact-source-node-id',),
    observation_policy='preserve', binding_digest=digest,
)
preview = apply_structural_candidate(source, plan, binding=binding)
manifest = preview.manifest  # archive separately, with the immutable source
replayed = replay_structural_candidate(source, manifest, binding=binding)
assert replayed.to_json() == preview.to_json()
```

`enumerate_structural_candidates` only proposes **explicitly selected** roots
and depths, in a fixed order, with a hard maximum (default 64, cap 128). It
never scans all nodes to choose a preferred branch, never fits a model, and
never silently truncates candidates. Hierarchy-only modes are omitted when the
source has no validated rooted hierarchy. A structurally valid proposal can
still be rejected at preview time if its chosen data policy would delete all
observed columns; the replayable *preview* is the authoritative validation.

## Validation / operational gates

```bash
PYTHONPATH=src pytest -q tests/test_su08_structural_candidates.py \
  tests/test_ml_uw_30_structural_transformations.py \
  tests/test_su05_structural_observation_binding.py \
  tests/test_su06_branch_uncertainty.py \
  tests/test_su07_branch_information.py
```

This patch does **not** authorize pruning based solely on low marginal MI,
low variance, or visually sparse graph branches. SU-09 must evaluate explicit
retention objectives on matched observations (and leakage-safe splits when
predictive). SU-10 will expose sandbox/overlays in the shared interactive UI.
