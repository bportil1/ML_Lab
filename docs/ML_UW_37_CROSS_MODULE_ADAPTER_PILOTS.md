# ML-UW-37 — Local cross-module structured-object handoff pilots

Status: **ML_Lab adapter implementation; no cross-repository changes**.

## Architectural boundary

ML_Lab consumes *explicitly supplied*, local JSON exports through its adapter
registry. It never imports a Code Analyzer or pyPIQUE Python module, parses a
repository implicitly, scans another module's directories, invokes external
services, or reads another module's SQLite database. The source module owns its
source code, models, databases, and export authorization. The common computation
engine continues to accept only validated `StructuredObject` instances.

ML-UW-37 adds an **opt-in** `cross_module_adapters()` registry. The original
`default_adapters()` registry and the ML-UW-27..35 computation interfaces are
unchanged. The ML-UW-36 project-local source selector offers the additional
artifact types without adding a new top-level UI tab. A source file is read
only when the user selects it; its path is scoped, its content fingerprint is
pinned, and subsequent actions recheck the fingerprint.

## Hierarchical AST handoff (Code Analyzer candidate)

**Handoff schema:** `ml-lab.hierarchy-handoff@1`

This is a newly defined ML_Lab interoperability envelope; **it is not a claim
that Code Analyzer currently emits this schema**. It is also reusable for any
other source that can provide an explicit tree/forest.

Example export (synthetic):

```json
{
  "schema": "ml-lab.hierarchy-handoff@1",
  "source_id": "code-analyzer:example:Example.java",
  "roots": ["unit"],
  "nodes": [
    {"id": "unit", "kind": "compilation_unit", "attributes": {},
     "source_ref": {"path": "src/Example.java", "line": 1}},
    {"id": "method", "kind": "method_decl", "attributes": {"name": "run"},
     "source_ref": {"path": "src/Example.java", "line": 2}}
  ],
  "edges": [
    {"id": "unit-method", "kind": "syntax_child", "parent": "unit",
     "child": "method", "ordinal": 0}
  ],
  "provenance": {"producer": "local-explicit-export"}
}
```

The adapter preserves node IDs, type labels, source references, attributes,
child ordinals, and provenance. It independently validates the declared roots,
acyclicity, and unique parent relations. It refuses cyclic/multi-parent input;
those inputs may instead use the existing `ml-lab.node-edge@1` generic graph
adapter if a tree representation is inappropriate. It does not infer runtime
observation rows from source nodes, AST locations, or repeated syntax patterns.

**Code Analyzer acceptance still pending:** no actual Code Analyzer repository
or exported AST file was supplied for ML-UW-37. The in-repository tests are
synthetic contract fixtures. The producing module must map its real AST output
to this envelope or an additional narrow adapter must be written and tested
against a real export. Do not claim a live Code Analyzer interface is connected.

## pyPIQUE concluded-model graph projection

Local source review: the supplied pyPIQUE archive contains
`src/pique_py/concluded_graph.py`, where `project_concluded_graph()` returns the
`pypique.concluded_graph` **schema version 1** envelope with a `graph` (role
`native_quality_model` or `recorded_signed_information_and_reporting`) and an
optional `learned_probabilistic_backing` (role
`immutable_learned_probabilistic_backing`). A separate module,
`shared_concluded_graph_pilot.py`, produces a display-only
`pah-shared.graph@1` view.

ML_Lab exposes the following explicit source choices:

| UI adapter | Input | Behavior |
| --- | --- | --- |
| `pypique_graph` | Concluded projection main `graph` | Imports all edges if they have uniform direction; refuses mixed-direction graphs |
| `pypique_graph_directed` | Concluded projection main `graph` | Imports only recorded directed edges, recording the number excluded |
| `pypique_graph_undirected` | Concluded projection main `graph` | Imports only recorded undirected edges, recording the number excluded |
| `pypique_backing` | Concluded `learned_probabilistic_backing` | Imports the recorded backing separately, not as a fitted ML_Lab probabilistic model |
| `shared_graph` | `pah-shared.graph@1` | Imports coordinates and presentation links; explicitly warns that scientific source detail is not in this display contract |

An MI/interpretability display may include directed signed information links
and undirected *display-only* attribution traces. The ML-UW-28 canonical
contract supports one directionality per graph object. **No implicit
conversion** is allowed. A mixed main graph requires explicit edge-direction
projection, and the resulting `AdaptationResult.losses` identifies excluded
edges. The original graph must remain available separately for complete
interpretation; selected filtered objects are not claimed to be lossless.

Saved acceptance values, learned weights, MI measures, and display trace flags
remain attributes with their original labels. No statistical observations are
fabricated from nodes, weights, or signed associations; neither a source edge
nor a learned connection is silently reinterpreted as a Bayesian-network CPT.
The imported model's archive/manifest hashes are **source claims**, not an
independent verification of the original archived concluded model. For trusted
scientific use, first produce the export through pyPIQUE's own verified
projection function; ML_Lab pins and verifies the exported JSON *file* bytes.

The pyPIQUE source code was available locally, but no genuine saved concluded
model/export was available in this patch's fixtures. Tests use representative
records matching the inspected producer shapes. End-to-end producer validation
with a real concluded model remains an acceptance item for ML-UW-38.

## Project-local UI use

1. Produce a JSON artifact through the owner module's explicit export API or
   adapt its public export into the documented hierarchy handoff envelope.
2. Place that export inside the active ML_Lab project artifact directory.
3. Select it under **Data → Structured data**, choose the matching adapter, and
   inspect reported topology, provenance warnings, and loss audit.
4. Use the existing Feature Profiling, Representation, Comparison, and General
   ML Tools workflows, subject to their scientific capability checks. An AST
   or graph with no aligned observation rows supports structural profiling,
   but MI/CMI or fitting requires a separately supplied valid sample set.

**Limits inherited from ML-UW-36:** The interactive selector caps JSON
artifacts at 2 MiB and validated observation rows at 10,000. Larger inputs
require the headless adapter API or a separately reviewed scale path.

## Acceptance and deferred work

- [x] AST-shaped tree and forest handoff contract, typed identity preservation,
      source-reference provenance, verified roots and ordinals.
- [x] pyPIQUE main versus backing kept separate, known schema/role validation,
      deterministic source-to-StructuredObject translation and loss audit.
- [x] Reusable PAH_Shared display graph contract and explicit metadata limits.
- [x] Existing domain-neutral analyses called on the outputs without modifying
      those algorithms or crossing repository data boundaries.
- [ ] Actual Code Analyzer AST export replay (source repo/export unavailable).
- [ ] Actual pyPIQUE concluded model export replay (model artifact unavailable).
- [ ] Flask/PAH graphical acceptance with installed optional UI dependencies.
- [ ] Large-graph/persistence/performance acceptance in ML-UW-38.

No files are automatically written by these adapters. The full-scientific
analysis loop remains **source artifact → adapter → validated StructuredObject
→ capability-gated methods**, with graph meaning and sample provenance explicit.
