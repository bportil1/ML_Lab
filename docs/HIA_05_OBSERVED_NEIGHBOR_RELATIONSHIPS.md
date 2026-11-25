# HIA-05 — Observed relationships among structural neighbors

**Repository owner:** standalone `ML_Lab`, not PAH host `modules/`.
**Schema:** `ml-lab.hia-neighbor-relationships@1`.
**Prerequisite:** HIA-01–04, SU-05 binding engine, and HIA-02 verified hierarchy.

## Scope and purpose

HIA-05 extends the HIA-03 source graph with an **opt-in observed-variable
comparison panel**. Its interface offers structural parent/child neighbors from
original source edges. Users **explicitly bind** the selected node and each
chosen adjacent node to a **distinct aligned observation column**. A separate,
optional node/column binding specifies a conditioning variable for CMI.

The structural edges are retained by ID and kind solely as selection/provenance
context. They **never** establish dependence, random-variable identity,
conditional independence, causal direction, or an information-flow claim.

## Headless API

```python
from ml_lab.structured import analyze_neighbor_relationships

report = analyze_neighbor_relationships(
    source, node_id='parent-id', variable_id='measured-column-A',
    neighbors=[{'node_id':'child-id', 'variable_id':'measured-column-B'}],
    conditioning=[{'node_id':'context-id', 'variable_id':'measured-column-C'}],
    occurrence_policy='preserve',
    bins={'measured-column-A': {'mode':'quantile', 'bins':4}},
    min_samples=5, min_stratum_samples=2,
)
```

A call is read-only and source-fingerprinted. `ObservationBindingPlan` and
`bind_observations` enforce SU-05 one-to-one mappings and repeated-cell policies.
For every selected neighbor, a separate pairwise sample population is used;
conditioning may reduce this sample population further. Every numerical result
reports its own support and omission status. Source **nodes are not samples**.

### Report fields

- `source_digest`, `binding_digest`, `binding_plan`, original node and column IDs
- `structural_context`, original edge IDs/kinds, and the explicit conditioning map
- `mutual_information.mi_bits`: empirical discrete **I(X;Y)** in bits
- `mutual_information.h_x_given_y_bits`: empirical **H(X | Y)** in bits
- `mutual_information.h_y_given_x_bits`: empirical **H(Y | X)** in bits
- `conditional_mutual_information.mi_bits`: empirical **I(X;Y | Z)** when Z declared
- conditional `h_x_given_y_bits`, `h_y_given_x_bits` are **H(X|Y,Z)** and **H(Y|X,Z)**
- `correlations.pearson`, `correlations.spearman`: pairwise complete, finite numeric rows;
  Spearman uses average midranks for ties
- per-estimate `status`, `support`, `excluded_rows`, and conditioning-stratum diagnostics

All calculations use observed **aligned records**, not an inferred joint or
conditional probability model. MI and conditional entropy use the existing
finite-sample discrete plug-in estimator primitives. Floats require explicit
quantile/equal-width bins *for information estimates*; numeric correlations do
not require binning. Integers, booleans, and strings are categorical for MI.

### Insufficient and unsupported cases

- No observed records or fewer than `min_samples` complete paired records: `insufficient_samples`.
- Repeated values under `preserve`: `unresolved_repeated_values`; use SU-05's
  `scalar_only` (reject) or explicit `mean_numeric` as appropriate. Because
  `mean_numeric` creates floating-point values, bin its output if used for MI.
- Missing/null/unknown cells are **never** zero-filled. Each pair uses its own
  complete cases, and CMI requires complete conditioning cells too.
- A conditional stratum smaller than `min_stratum_samples` yields
  `insufficient_conditioning_support`, not a guessed CMI.
- Unsupported JSON-valued cells and unbinned floats produce an unavailable
  information estimate with a reason; correlations can still be available.
- Non-numeric cells, constant values, and insufficient numeric support produce
  explicit correlation statuses. No p-values or significance claims are made.
- Any mapping with duplicate columns, duplicate nodes, unknown columns, or
  a neighbor not adjacent to the selected original node is rejected.

## UI and integration

Data → Structured Uncertainty → **Explore source graph** → click an original
node → **Neighbor relationships**. Choose its observed column, pick any directly
adjacent node, choose that node's column, and click **Add explicit neighbor
mapping**. Repeat as needed. Optionally declare a conditioner node/column and
numeric binning JSON. Press **Compute mapped neighbor statistics** to request a
fresh SHA-checked report. A results table shows MI, conditional entropies, CMI,
Pearson/Spearman, and support; JSON export includes source evidence and all
warnings. Node switching and source reopening invalidate previous results.

The existing graph, hierarchy navigator, HIA-04 atomic profile, network
analysis, and SU-10 sandbox are unchanged. This patch also **loads HIA-04's
atomic JavaScript controller**, which was fingerprinted but omitted from the
HIA-04 template's script tags.

## Performance and boundaries

No fixed node/edge display limit is reintroduced. The view can select any
original neighbor; only the **explicitly requested comparisons** are computed.
Each pair scans aligned observations independently. SU-05's separate
observation-row bound and the route's 64-KiB request guard remain defense-in-
depth against unbounded computation. This is *not* a level estimator and does
not assume conditional independence of earlier structural levels. Branch and
level statistics remain HIA-06–09 work.

## Acceptance

Run in the standalone checkout:

```bash
PYTHONPATH=src python -m pytest -q tests/test_hia05_neighbor_relationships.py
node --check src/ml_lab/ui/static/ml_lab_hia_neighbors.js
```

Use the PAH setup lifecycle and restart the ML_Lab process after applying.
Live hosted browser acceptance requires opening a real aligned source and
verifying node mapping, network navigation, optional CMI, JSON export, and
source-revision invalidation.
