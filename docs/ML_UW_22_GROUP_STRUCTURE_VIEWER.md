# ML-UW-22 — Graphical XML structure for individual candidate groups

## Contract

The XML-C2 collection page now offers a read-only, selectable hierarchy for **each
candidate group**. It uses the **same** `ml-lab.xml-collection-union-structure@1`
artifact produced by ML-UW-19. No source XML documents are loaded by the browser
and no new parser, recipe-approval path, persistent schema or external service
is introduced.

The viewer uses canonical, JSON-Pointer-escaped expanded QName paths as stable
node identities. Element and attribute field nodes display their namespaces,
observed occurrence counts, group-relative source presence, and missing-source
counts. Elements additionally show source-level optionality, repetition and mixed
content. The default hierarchy renders roots only, expands branches on demand
(in batches of 80 siblings), and provides a bounded path/name search. Selecting
nodes shows their canonical path and source evidence. The group selector and
per-row “View structure” buttons display one group's structure at a time.

The raw machine-readable union C2 JSON remains available in an expandable
section, and is the single serialized source of truth for the viewer. The
rendered DOM is built with `textContent` rather than unsanitized XML/HTML.

## Interpretation and boundaries

- Counts and source memberships are **observations**, not imputed records or
  inferred null values. In particular, missing-source counts do not measure
  within-source missing record values.
- Group membership remains advisory and does **not** automatically authorize
  shared recipes, imports, or exports. C3/C4 continue to apply their existing
  review/validation gates.
- Search is a bounded navigation aid, not an authoritative query of source
  values. Source filenames and canonical names come from the C2 report.
- Expanded hierarchy nodes are rendered lazily, but XML-C2 comparison is still
  synchronous and computes the complete observed union before this page loads.
- Per-source extraction and the existing single-file XML hierarchy viewer
  are unchanged. ML-UW-23 will use the same observed union for the **across-group**
  consolidated graphical structure.

## Verification

Test the native helper with mixed namespace prefixes, optional elements and
attributes, source isolation, and root partitions. Verify no raw XML fetches or
unsafe `innerHTML` use in the viewer, and run the optional Flask route test in
an installed ML_Lab/PAH environment. Open an XML collection with Compare checked,
select different candidate groups, search for namespaced paths, expand branches,
inspect attributes, and ensure existing C3/C4 actions remain available.
