# ML-UW-29 — Source adapters and normalization

**Baseline:** `ML_Lab-main(5).zip` + ML-UW-28.  **Status:** First source-adapter implementations; no scientific estimators or UI changes.

## Scope and public API

```python
from ml_lab.structured.adapters import default_adapters, JSONAdapter

registry = default_adapters()  # local registry, never a process-global singleton
json_result = registry.adapt('json', {'items': [1, None, 2]})
json_result.object.require('hierarchy_summary')

# Optional strict JSON-text entry point; rejects duplicate keys, NaN/Infinity.
json_result = JSONAdapter().from_text('{"items":[1,null,2]}')

# XML uses the versioned, previously inspected structure artifact, not raw XML.
from ml_lab.data.xml_ingest import inspect_xml_structure
xml_structure = inspect_xml_structure('/path/approved.xml')
xml_result = registry.adapt('xml_structure', xml_structure)

# AST/network producers export generic nodes and edges through APIs/artifacts.
graph_result = registry.adapt('node_edge', {
    'schema': 'ml-lab.node-edge@1',
    'topology': 'dag',
    'nodes': [{'id': 'a', 'kind': 'node', 'attributes': {}},
              {'id': 'b', 'kind': 'node', 'attributes': {}}],
    'edges': [{'id': 'ab', 'kind': 'link', 'source_id': 'a', 'target_id': 'b'}],
})
```

The registry accepts any adapter implementing `adapt(source) -> AdaptationResult`; adding a new source type requires no change to the `StructuredObject` core, analysis contracts, or other adapters. The registry and adapters perform no implicit network, database, or file IO. Reading XML through the **existing safe parser** is an explicit caller action.

`AdaptationResult` contains a validated `StructuredObject`, a declared source schema, `warnings`, and `losses`. `require_lossless()` fails closed if information has been discarded or was not contained in the source artifact. These diagnostics are programmatically inspectable; they are not silent log messages.

## Individual adapters

| Adapter | Input | Structural semantics | Observation semantics | Preservation / limitations |
| --- | --- | --- | --- | --- |
| `json` | Parsed JSON value or explicit strict JSON text | True instance tree; JSON-Pointer node IDs; stable sorted object keys; array-edge ordinals | No aligned observations synthesized from a single document | `null` = explicit null, `""` = empty, scalar values preserved. JSON object key order is not semantically ordered. Node/depth caps are enforced. |
| `xml_structure` | `XmlStructureArtifact` or its versioned `to_record()` mapping | Path-level tree, including separately typed attribute nodes; expanded namespace-aware QNames and aggregated occurrence profiles | No fabricated record-level observations | **Lossy with respect to raw source:** source structure profiles do not contain individual element instances, occurrence-specific sibling order, or original record values. The full profile and source summary remain in metadata/provenance. |
| `node_edge` | Canonical `StructuredObject` or `ml-lab.node-edge@1` mapping | Explicit tree, forest, DAG, directed/undirected graph; existing edges, types, ordinals, multiedges preserved | Explicit aligned cells passed to core validator; no automatic pairing | Unknown fields rejected rather than silently discarded; topology declarations/capabilities validated against actual edges. |

`xml_structure` **cannot certify that XML source content remains unchanged** merely by reading a stored artifact. It retains the artifact's source hashes as *claims*. Source owners must verify current source identity before producing/publishing the artifact, and must verify again if a later step needs fresh source bytes. No reads from pyPIQUE, Code Analyzer or other project SQLite stores are permitted.

The JSON adapter's content hash is computed from a canonical serialization of the actual supplied value. The source schema/provenance of graph imports is preserved as metadata, not converted into probabilistic assumptions.

## Scientific safety contracts

- A source graph is **not** automatically a statistical association graph or a probabilistic graphical model.
- Tree-only operations require validated `rooted_hierarchy`, `unique_parent`, and `acyclic` capabilities. DAGs with multiple parents and cycles cannot be silently coerced to trees.
- Aligned observations are supplied explicitly. Neither an XML source profile nor a single JSON document is enough to infer paired statistical samples.
- Missing value, empty string and explicit null are distinct. Missing values from optional XML paths are not fabricated as observed values by the profile adapter. Null provenance remains an adapter/observation concern.
- Limits on JSON nodes/depth and XML source ingestion remain explicit; there is no background recomputation, dataset activation, or source mutation.

## Acceptance and rollback

`PYTHONPATH=src pytest -q tests/test_ml_uw_27_structured_contract.py tests/test_ml_uw_28_structured_object.py tests/test_ml_uw_29_source_adapters.py` (adjust ML-UW-27 test filename to the checkout's existing test naming if needed). XML namespace aliases, nulls, arrays, AST-shaped graphs, DAG/cyclic network fixtures, claim spoofing, malformed profiles and resource bounds are covered.

This patch only adds `src/ml_lab/structured/adapters/`, tests, and this document. No requirements, SQLite migrations, UI routes, generated artifacts, existing XML extraction behavior, or scientific training engines are changed. It can be reverted without data migration.

**Next: ML-UW-30** will add explicit, auditable full/partial/flat structural transformations on validated objects. No source-specific transformations belong in the computational core.
