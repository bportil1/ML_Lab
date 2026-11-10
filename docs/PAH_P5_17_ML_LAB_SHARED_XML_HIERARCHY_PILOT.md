# P5-17 — ML_Lab XML → PAH_Shared hierarchy pilot

## Scope

`ml_lab.data.xml_shared_hierarchy.project_xml_structure_hierarchy` accepts an existing
`XmlStructureArtifact` (`ml-lab.xml-structure@2`) and returns a `model` compatible
with `pah-shared.hierarchy@1` plus a separate XML `provenance` map.

- This is **opt-in, read-only, and Python-only**. It does not change any existing
  XML inspector route, tree navigation, record selection, extraction, recipe,
  dataset registration, or sqlite catalog.
- Shared display IDs are **unchanged canonical XML paths**. JSON-pointer escaping
  of expanded QName segments is preserved; no derived/shortened keys.
- Node order remains the authoritative structure profile order. Parent links
  come from canonical path structure. Sibling ordering is not inferred anew.
- The presentation model contains only `schema`, `title`, and `nodes` with
  `id`, `parent_id`, `label`, `kind`; it deliberately does not encode element
  counts, namespaces, candidate scores, source paths, or scientific values.
- The sidecar `provenance` retains source fingerprint, source hash, canonical
  and display paths, expanded tags, cardinality flags and occurrence counts.
  It remains **read-only context**, not a storage or authorization contract.

## Limits and non-goals

Graphs with >5,000 unique paths, nesting >64, labels >240 characters, or
canonical path identifiers >96 characters are rejected rather than truncated.
Unsupported structures remain renderable in the existing native XML explorer.
No UI integration is claimed; consumer-side browser parity and host tests remain
P5-17/P5-20 acceptance gates. `PAH_Shared` is not imported at Python runtime.
