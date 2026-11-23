# ML-UW-28 — Canonical `StructuredObject` (implementation and acceptance)

**Baseline:** `ML_Lab-main(5).zip`, with ML-UW-27 already present.
**Scope:** Strict, immutable, schema-versioned representation and topology validation.
**Next:** ML-UW-29 source adapters for XML, JSON and generic graph artifacts.
**Not implemented here:** XML parsing/conversion, structured uncertainty estimators, probabilistic inference, persistence/UI changes, schema migrations or automatic artifact activation.

## Public API

```python
from ml_lab.structured import StructuredObject

obj = StructuredObject.from_dict(canonical_mapping)
obj2 = StructuredObject.from_json(obj.to_json())
assert obj2.to_dict() == obj.to_dict()

obj.require('hierarchy_summary')  # validated structural preconditions
obj.require_aligned(['variable_a', 'variable_b'])  # explicit paired cells
```

- The only construction path validates all fields. The result is deeply immutable: nested mappings are read-only and lists become tuples. `to_dict()` returns a detached, mutable JSON-compatible copy.
- `to_json()` is deterministic (sorted mapping keys, UTF-8-compatible text, no `NaN`/`Infinity`); JSON import rejects duplicate keys and unknown schema versions/fields.
- `schema` is `ml-lab.structured-object@1`. A caller may supply a nonempty `object_id`; otherwise the implementation derives one using SHA-256 over the canonical JSON representation of the **source graph, observations and provenance**, excluding claimed capabilities. This establishes a reproducible object identity, **not** a signed attestation that the original source file is unchanged. Source integrity/fingerprinting remains the adapter's responsibility.
- The object contains one source-topology kind, nodes, edges, observation unit, aligned observations and opaque provenance. Source-specific interpretations belong in typed fields or metadata; the computational core contains no XML/AST/pyPIQUE branching.
- The `topology.capabilities` field may be omitted at import. When supplied, its contents must match the validator's **derived** results exactly; missing, forged, duplicated or unsupported claims fail closed. The serialized object always includes the derived capabilities.

## Validation semantics

| Property | Rule |
| --- | --- |
| Nodes and edges | Nonempty unique IDs; edges reference existing node IDs; repeated edges are allowed only with distinct edge IDs |
| Direction | Tree, forest, DAG and directed graph have directed edges; undirected graphs cannot claim directed edges |
| Tree | Exactly one root, acyclic, at most one incoming edge per node |
| Forest | One or more roots, acyclic, at most one incoming edge per node |
| DAG | Acyclic directed graph; multiple parents are allowed |
| Directed graph | Cycles allowed; a cycle never implies a rooted hierarchy or unique-parent capability |
| Undirected graph | Acyclicity computed using graph connectivity; self-loops and parallel edges form cycles |
| Ordered children | Derived only when every directed edge has a nonnegative ordinal, unique among siblings under the same source |
| Observation alignment | Nonempty set of rows with unique observation IDs, a common observation-unit declaration, identical explicit variable keys, valid status/value cells; no implicit fill for missing variables |
| Nulls and multiplicity | `missing`, `explicit_null`, `unknown` have null values; `empty` has `""`; `multiple` uses a nonempty ordered occurrence list and null summary value; malformed/parse-error states are not silently converted to null |

The validator derives `graph_edges`, `directed_edges`, `acyclic`, `unique_parent`, `rooted_hierarchy`, `ordered_children` and `aligned_observations` only where justified. It never derives `explicit_probability_model`, because that requires a **separate probabilistic model artifact** in later sprints.

**Important distinction:** `aligned_observations` proves only that rows contain explicit comparable variable identities; it does **not** establish statistical independence, adequate sample size, a valid entropy estimator, or an identified observation distribution. Graph topology is not itself a probability model. Algorithms in ML-UW-31 through ML-UW-34 must apply their own additional scientific validation.

## Synthetic acceptance evidence

The four pre-existing ML-UW-27 fixtures are now valid schema payloads: their test-only labels were moved under `provenance.fixture_id`. Each loads directly and round-trips without a special case in the runtime.

- `xml_tree.json`: generic rooted ordered hierarchy; explicitly empty, missing and null cells retained.
- `ast_tree.json`: generic rooted ordered hierarchy with no language-specific runtime branch.
- `shared_child_dag.json`: multi-parent acyclic graph; tree-only operations denied.
- `cyclic_network.json`: directed cycle; hierarchy and information-analysis gates denied when relevant prerequisites are absent.

Negative tests cover capability spoofing, dangling/duplicate IDs, bad ordinals, direction mismatch, cyclic DAGs, invalid roots, duplicate/unaligned observations, absent-value ambiguity, repeated occurrences, JSON schema/format errors, and deep-copy immutability.

## Security, artifact ownership and rollback

ML-UW-28 is a **pure in-memory module**. It reads no files on construction, writes no runtime output, accesses no database, creates no network connections, and does not modify PAH integration or the experiment schema. The ML-UW-29 adapters will validate source fingerprints against artifacts owned by their source module/project; a `source_fingerprint` value in provenance is a *claim*, not reverified file content at this layer. This patch is safely reversible without data migration. Existing XML processing, model fitting, and UI behavior remain unchanged.

Acceptance of the object contract is separate from PAH/browser acceptance. No complete hierarchical uncertainty computation is claimed at this stage.
