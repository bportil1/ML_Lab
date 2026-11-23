# ML-UW-27 — Structured uncertainty: architecture audit and scientific contract

**Status:** Contract and synthetic fixtures only; no structured analysis algorithm or source adapter is implemented in this increment.
**Predecessor:** ML-UW-26 (unified UI/grid and project-local artifact preflight).
**Follow-on:** ML-UW-28 (`StructuredObject` validator, serialization, persistence integration), then ML-UW-29 adapters.
**Scope:** Descriptive analysis (A), information theory (B), explicit hierarchical probabilistic models (C). Learned topology, GNNs and structured neural autoencoders remain deferred.

## Design boundary

All future structural computation will consume a **validated, versioned, domain-independent object plus explicitly aligned observations**. Source-specific names, file parsing, and semantic interpretations belong to adapters. Neither a source parent-child link nor an MI-weighted association edge is automatically a factor in a probabilistic model.

Data must remain owned by the originating module/project. A Code Analyzer AST or pyPIQUE graph is imported through a **published API or export artifact**, never through ML_Lab reading another module's private SQLite file. Production code must not perform network access or silently process data when opening a UI page.

## Audit of actual ML-UW-26 baseline

| Existing code | What it already provides | What ML-UW-28+ must **not** assume |
| --- | --- | --- |
| `data/xml_similarity.py`: `compare_xml_collection`, `_observed_union`, `UNION_SCHEMA` | Canonical source-linked path union, group membership, path/attribute counts, root partitions, optional-path references | Path union is not yet a per-document instance graph; compatibility groups do not authorize probabilistic edges or merging |
| `data/xml_null_provenance.py`: `catalog_xml_null_fields`, `preview_xml_null_provenance` | Canonical expanded-QName field IDs, per-record condition and repeated-observation addresses | XML-specific status names are **adapter provenance**, not generic numeric values or probability estimates |
| `data/xml_typed_representation.py`: typed-variable representation | Distinguishes scalar, object, list, content/path token and linked-child-record representations | No generic graph `StructuredObject` contract currently exists |
| `data/xml_collection_recipe.py`, `xml_collection_columns.py` | Reviewed recipes and canonical column identity across prefix aliases | An export column name is not a stable node identity; materialization is an explicit transformation |
| `data/xml_null_export.py`, `xml_collection_export.py`, `xml_collection_scale.py` | Round-trip-aware `\N`/`\E` delimited export, `Comments` provenance, matching C5/C7 paths | Serialized `Comments` is not a numeric feature or the authoritative internal null-reason record |
| `experiments/workbench.py`, `experiments/datasets.py` | Active, revision-pinned project-local dataset; fingerprint and ownership verification | A dataset is not implicitly activated by reading or visualizing it; outputs must not change another tab's workbench generation |
| `experiments/store.py`, `experiments/artifacts.py` | Project-owned SQLite schema v3 and hashed artifact references with explicit run provenance | There must be no global cross-module database and no schema migration in ML-UW-27 |
| `experiments/workbench_preflight.py`, `feature_profile_cache.py` | Read-only artifact discovery, verification and bounded summary reuse | “Previously computed” is not proof of scientific validity unless source, specification and estimator match |
| `data/feature_explorer.py`, `representation/comparison_workbench.py` | Tabular descriptive/dependence statistics and matched flat-embedding comparison | Neither method is a hierarchy-aware uncertainty estimator; raw entropies in different spaces are not directly comparable |
| `ui/static/ml_lab_tables.js`, XML group viewer | Shared tabular styles and source-linked group tree inspection | The browser viewer is not the statistical engine and must not be the only representation of topology |

**Current gaps:** No format-neutral node/edge/observation object; no generic capability validator for concrete topology; no adapter contract; no observation-alignment standard for information estimators; no explicit hierarchy-to-PGM mapping. ML-UW-27 introduces only a minimal, dependency-free **vocabulary and fail-closed precondition gates** in `ml_lab.structured.contracts`, plus acceptance fixtures. It does **not** implement these gaps prematurely.

## Proposed version-1 object contract (implementation in ML-UW-28)

`schema = "ml-lab.structured-object@1"` with the following components:

- `object_id`, `schema`, `provenance`: stable object identity, schema version, originating source fingerprint/adapter version, optional project-local artifact references.
- `topology.kind`: exactly `tree`, `forest`, `dag`, `directed_graph`, or `undirected_graph`. Other topology classes require an explicit future schema extension.
- `topology.capabilities`: **validated** flags, not a user-controlled shortcut. See vocabulary below.
- `nodes[]`: unique stable `id`, typed `kind`, source-reference metadata, typed `attributes` and optional display labels. The source's tree path can be recorded here, but is not required or hardcoded.
- `edges[]`: unique `id`, endpoint IDs, typed relationship, optional ordinal (ordered siblings), direction and edge attributes. Repeat occurrences/multiedges require distinct IDs; no implicit deduplication.
- `observation_unit`: one of `source`, `record`, `node`, `graph`, defined per study/observation set. Explicit unit identity prevents flattening from multiplying independent samples.
- `observations[]`: one `observation_id` per sampling unit with values keyed by stable variable identity; each cell has a **generic** `status` and `value` and optional source-specific `source_reason`, occurrence information and lineage. Explicitly null entries remain present in every row's `values` mapping.
- Separate optional **association graph** and **probability-model specification** artifacts may reference canonical IDs; neither may overwrite the source object's edges. They will have their own schemas and estimators.

The four JSON files in `tests/fixtures/structured_uncertainty/` are **illustrative contract fixtures**, not production `StructuredObject` instances and not proof that a source adapter exists. ML-UW-28 must validate and round-trip them with no domain-specific branches.

### Stable vocabulary introduced now

| Contract | Allowed values or operations | Interpretation |
| --- | --- | --- |
| Topology | `tree`, `forest`, `dag`, `directed_graph`, `undirected_graph` | Declared structural family, **to be validated** by ML-UW-28 |
| Observation unit | `source`, `record`, `node`, `graph` | Explicit unit on which rows are aligned; a graph's nodes alone do not create a sample |
| Generic cell states | `present`, `empty`, `missing`, `explicit_null`, `unknown`, `multiple` | Source-agnostic status; `missing`/`explicit_null`/`unknown` have `None` values, `empty` has `""`; `multiple` needs explicit occurrences |
| Source null reasons | e.g. `path_absent`, `attribute_absent`, `explicit_xsi_nil`, `unresolved_null` | **Metadata from an adapter**, not a finite source-neutral core enumeration |
| Capabilities | `graph_edges`, `directed_edges`, `acyclic`, `rooted_hierarchy`, `unique_parent`, `ordered_children`, `aligned_observations`, `explicit_probability_model` | Declared prerequisites. `require_capabilities` checks presence; *concrete truth* must be validated later |
| Core operation gates | `graph_summary`, `hierarchy_summary`, `depth_reduction`, `conditional_information`, `hierarchical_probability_inference` | Minimum requirements only; finite-sample, estimator and model-specific gates will be added with their algorithms |

Examples of **adapter mappings**: an absent XML field becomes `{status: "missing", value: null, source_reason: "path_absent"}`; an explicit `xsi:nil` becomes `{status: "explicit_null", value: null, source_reason: "explicit_xsi_nil"}`; an empty element becomes `{status: "empty", value: ""}`. These are examples, **not XML branches in the computational core**. Malformed XML is a source-validation error, **not an observation with a synthetic null**.

### Topology and eligibility rules

- Tree/forest require directed acyclic parent-child relationships and at most one parent per node. Trees have one root; forests may have several. Ordered children must preserve explicit ordinals where supplied. Validate disconnected components, absent endpoints, duplicate IDs, repeated edges and cycles.
- DAGs may have multiple parents. A cycle invalidates `acyclic`. General directed graphs may have cycles, and neither a graph nor a DAG automatically satisfies `rooted_hierarchy` or `unique_parent`.
- A graph representation may contain multiple distinct edge *kinds*; a domain-specific hierarchy edge, a measured association edge and a probabilistic factor edge must never silently become interchangeable.
- Algorithms shall reject missing prerequisites with actionable error messages, not prune edges, pick an arbitrary parent, fill an unavailable parameter with zero, or re-label a cyclic graph as a tree.
- `require_capabilities()` is a **contract gate**; it cannot establish that a caller's capability claims are true. ML-UW-28 must calculate/validate capabilities against the actual object. Likewise, `require_observation_alignment()` verifies explicit pairs and normalized cell shapes, not statistical independence, estimator consistency or adequate support.

### Sampling, uncertainty and model assumptions

- A source (file), record, node or graph can be an observation unit; the user or experiment must identify the unit. It must remain unchanged across flat/partial/full alternatives unless an explicit remapping is documented.
- MI, conditional MI and entropy require **aligned samples**, a documented missingness policy and a declared estimator. There is no valid MI estimate derived solely from an AST edge or a pyPIQUE link weight. A pyPIQUE graph may supply an **existing estimate** with provenance, which differs from estimating a new quantity from observations.
- Option A measures structural coverage, cardinality and within-/between-level variability with denominators. A missing optional branch is not automatically zero variance or evidence of irrelevance.
- Option B must report estimator, discretization, seed, effective support, confidence/uncertainty method and exclusion policy; small-sample and repeated-observation limitations must be explicit.
- Option C requires a **separate explicit factor graph** with validated conditional distributions, variable mappings, fit/query bounds and a documented inference mode. Source hierarchy alone never selects a PGM.
- “Information preserved” must refer to the **same target, observation unit or reconstruction space**. Never subtract raw entropies across differently dimensional feature spaces and call it information loss.

## Fixtures and acceptance gates

| Fixture | Intended invariants | Expected capability failures |
| --- | --- | --- |
| `xml_tree.json` | One root, ordered descendants, explicit missing vs empty vs nil; aligned record observations | No implicit probability model |
| `ast_tree.json` | One-root directed tree, ordered child references; aligned source observations | No AST-specific core calculation |
| `shared_child_dag.json` | Two parents for one node, no cycles; aligned graph observations | No tree-only depth reduction |
| `cyclic_network.json` | Directed cycle; no aligned observations supplied | No DAG/PGM inference and no tree-only hierarchy operation |

**ML-UW-27 exit gate:** All synthetic fixtures parse, IDs are unique, edge endpoints exist, declared topology matches the test structure, null/empty values remain distinct, sample alignment rejects missing/duplicate observations, and operation gates fail closed. No change to existing extraction, grouping, ML algorithms, UI routes, project-local SQLite version or stored artifacts.

**ML-UW-28 entry gate:** Implement an immutable/validated structured object, strict serialization and deterministic source identities; validate and *derive* topology capabilities rather than trust claims. Add true project-local artifact registration only after schema and adapter-level provenance are established.

## Rollback, deployment and security

This patch introduces `ml_lab.structured` vocabulary, synthetic JSON fixtures, tests and this internal note. It does not migrate SQLite, write runtime artifacts, start jobs, change Flask routing, or import Code Analyzer/pyPIQUE. Revert it without altering existing experiment data. Headless test success is not a Flask or PAH end-to-end acceptance claim. Keep future generated artifacts outside repository worktrees where practical and update scoped `.gitignore` rules as part of the already-prioritized repository audit.
