# ML-UW-30 — Structure-preserving and reductive transformations

**Baseline:** `ML_Lab-main(5).zip` plus ML-UW-28 and ML-UW-29. **Implementation:** `ml_lab.structured.transformations` only. No UI, SQLite, training, source parsing, or workflow activation changes.

## Scientific contract and public API

```python
from ml_lab.structured import TransformationPlan, transform

full = transform(structured_object, TransformationPlan('full'))
partial = transform(structured_object, TransformationPlan('depth', max_depth=2))
subtree = transform(structured_object, TransformationPlan('subtree', roots=('node-id',)))
summary = transform(structured_object, TransformationPlan('aggregate', max_depth=1))
flat = transform(structured_object, TransformationPlan('flat'))
node_table = flat.flat_tables()['nodes']
observation_table = flat.flat_tables()['observations']
report = partial.audit  # detached copy with full source-to-output node lineage
```

`TransformationPlan` is a frozen, validated, versioned specification (`ml-lab.structure-plan@1`) with deterministic `to_json()` / `to_dict()` and strict `from_dict()` validation. Every call returns a validated immutable `StructuredObject` and a versioned detached audit (`ml-lab.structure-transformation@1`): source/output object IDs, plan, retained/dropped node and edge IDs, node identity mapping, collapsed ancestry, retained/dropped aligned observation variables, loss categories, and deliberately conservative reversibility.

### View modes

| Mode | Topology and data semantics | Limitations / loss |
| --- | --- | --- |
| `full` | Returns the input object unchanged (when no explicit feature selection) | Identity is reversible; no source-file re-verification implied |
| `subtree` | For validated rooted tree/forest, retain complete descendants of explicitly selected node IDs; one selected root yields tree, disconnected selections yield forest | Other branches and their relationships/attributes are dropped, with IDs recorded |
| `depth` | For validated rooted tree/forest, preserve nodes to absolute depth from root(s), with edges connecting retained nodes | Deeper node attributes and relationships are removed and reported |
| `aggregate` | Depth reduction plus *structural* summaries on each retained boundary node: removed-node/edge counts and node-kind counts | **Does not aggregate values**, merge observations, infer a distribution, or fabricate a new sample. Summaries are not invertible |
| `flat` | Retain node identities/attributes, discard edges explicitly, produce a graph with no asserted hierarchy and separate node/observation tables | Relationship semantics are lost; resulting rows are **not** a learned feature matrix or an automatic node-observation join |

DAGs, general directed or undirected graphs, and cycles may use `full` or `flat`. Hierarchical `subtree`, `depth`, and `aggregate` **require verified** `rooted_hierarchy`, `unique_parent`, and `acyclic` capabilities; the engine does not choose a spanning tree, remove a cycle, or assign arbitrary parents.

### Observation and provenance invariants

- **All modes preserve the original observation count, `observation_id` order and observation unit.** Structural edges never create new statistical samples; node count and observation count must not be conflated.
- Aligned cells retain their exact `status`, `value`, `occurrences`, `source_reason`, lineage, and metadata. Source-specific meanings remain opaque; the engine knows nothing about XML/AST/pyPIQUE.
- To explicitly project aligned observation columns, pass `variables=('stable-variable-id', ...)`. All rows must already be aligned and contain the requested variables. Projection does **not** infer mappings from node IDs, flatten repeated values, or perform cross-table joins. Removed variables are listed in the audit.
- The flat representation offers two independent tables, one for nodes and one for existing observation records. If a study needs a domain-specific relation or aggregation between these, it must supply a separate explicit, validated mapping in a later operation.
- For `aggregate`, only structural counts are summarized. Dropped node attribute values cannot be reconstructed from these counts. The audit records every dropped identity. `lossless` and `reversible` are not used as synonyms; only unchanged `full` mode is certified reversible from its output alone.
- Every non-identity output receives transformation provenance referencing the input object ID and plan. Validation derives new capabilities from the resulting edges. Output IDs are deterministic for the same source and plan; the operation performs **no filesystem reads, writes, source verification, database access, network requests, or experiment activation**.
- Underlying sampling-group leakage cannot be ruled out by a transformation: no train/test splitting occurs here. Later Options A–C must define matched evaluation units and provenance-aware splits.

### Acceptance, known boundaries, and next steps

`PYTHONPATH=src pytest -q tests/test_ml_uw_27_structured_contract.py tests/test_ml_uw_28_structured_object.py tests/test_ml_uw_29_source_adapters.py tests/test_ml_uw_30_structural_transformations.py`

Coverage includes JSON-shaped hierarchies, the existing XML-shaped and AST-shaped fixtures, overlapping and disjoint subtree roots, forest topology, depth boundary counting, deterministic structural aggregation, DAG/cycle rejection of tree-only operations, preserved null provenance, explicit feature projection, and stable observation counts.

**Not implemented in this sprint:** XML/JSON record-level schema harmonization, numerical aggregation of values, structure-aware feature learning, flattened source-specific joins, predictive-utility scoring, estimator fitting, UI wiring, or storage migrations. The next sprint (ML-UW-31) adds **Option A** descriptive hierarchical uncertainty, with explicit support denominators and treatment of missingness. The existing generated-output and `.gitignore` audit remains separately prioritized.
