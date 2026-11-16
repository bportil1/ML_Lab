# ML-UW-23 — Consolidated XML union viewer

## Scope

The XML-C2 collection results now provide **two separate read-only views**
backed by the *same* `ml-lab.xml-collection-union-structure@1` JSON payload:

1. **Individual candidate group** (ML-UW-22): only paths observed within that group.
2. **Entire XML collection** (ML-UW-23): every canonical element/attribute path
   from every compared source, including groups with different root tags.

The views share their JavaScript tree renderer, lazy sibling pagination (80 at
once), field inspector, path search, and safe text-only DOM construction. No
new XML parser, browser fetch, persistent storage, dependencies, source rewriting,
or automatic recipe approval is introduced.

The consolidated viewer shows a searchable, expandable hierarchy with a root
partition selector, source coverage, source-level optionality, repetition,
attributes and namespace-aware field identities. Selecting a field generates a
**group-by-group coverage table**, including groups where the path is completely
absent; it renders 25 group rows at a time to keep the inspector responsive.

## Interpretation and limitations

- Coverage is derived only from **verified and compared** source inventories.
  Excluded or changed files are not included. A source absent from a path is
  not the same as a null-filled record in an extraction output.
- A root filter restricts what is *displayed*, not the global denominators:
  `source_count` and the group coverage breakdown describe the complete union.
- Group membership, candidate record roots, and union paths are **observational**.
  They do not bypass C3/C4 review, source-integrity validation, or shared-recipe
  authorization; incompatible roots can remain in separate root partitions.
- A valid union must partition its verified sources into exactly one group
  each. The viewer declines to make up a coverage table if membership is
  inconsistent.
- The viewer still depends on the synchronous XML-C2 computation; rendering is
  lazy, but large input comparisons themselves are not streamed.
- No inferred missing values, null provenance, or `Comments` fields are created
  by the viewer: these remain properties of ML-UW-20/21 recipe/preview/export.

## Acceptance checklist

- Compare a collection with mixed QName prefixes, two root namespaces, missing
  paths, attributes and repeated elements; check group and union trees agree.
- Search canonical namespaced paths, filter root partitions, inspect each
  field's group-by-group presence and optional counts, and expand batched nodes.
- Check that neither viewer fetches raw XML, that group/union tree switching
  preserves C3/C4 buttons, and that the raw union JSON remains available.
- Run Flask/PAH browser integration tests on the actual host installation;
  local headless JavaScript tests do **not** substitute for this.

Rollback: remove ML-UW-23 changes only; ML-UW-22 group viewer and ML-UW-21
null-aware exports remain unchanged.
