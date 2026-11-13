# ML-UW-19 — Null-tolerant XML grouping and observed union (internal)

## Scope and dependency

This increment follows ML-UW-18's namespace-aware field inventory and null-provenance APIs. It changes **C2 comparison and complete-link candidate grouping** and adds a read-only `union_structure` to the existing `ml-lab.xml-collection-similarity@1` report. It **does not** select record roots for extraction, authorize merges, change C4 recipes, materialize nulls, or modify C5–C7 datasets/exports. The group and consolidated graphical views follow in ML-UW-22/23.

No persistent DB schema, source XML, parsing safety limits, or dependencies change. The serialized C2 report retains its existing schema and keys; the `union_structure` and additional pairwise diagnostic flags are additive. Tools reading existing keys continue to work.

## Compatibility rules (C2 → C3)

- Expanded root QName differs: **incompatible** (must not be grouped automatically). Different namespace prefixes with the same expanded QName are equivalent.
- Repeated-element structure differs on a **shared** path: **review_required**. An array/scalar mapping is not silently selected.
- Mixed-content structure differs on a **shared** path: **review_required**; the engine cannot assume a scalar field mapping.
- No common automatically detected record-root evidence: **review_required** (unless identical). The sources may still appear in one **review-required candidate group**, avoiding artificial fragmentation of single-record XML, but this group cannot auto-prepare a recipe. C3 acknowledgment and a C4 record root present in every selected source are still required.
- Paths or attributes present in only some sources, absent/empty values, observed text-presence differences, and per-source optionality differences: **compatible_variant_candidate**, provided no harder conflict exists and the detected record-root evidence agrees.
- Pairwise path/attribute Jaccard and the user-specified similarity reference are retained as **advisory evidence**. A low Jaccard score caused by missing paths does not force fragmentation. `below_similarity_threshold` records that fact. The threshold is no longer an approval cutoff for missing-only differences.
- Existing complete-link candidate grouping remains in place. C3 still requires acknowledgment for variant pairs, checks every selected pair (no sampling), and does not grant merge authorization. C4 continues to validate selected sources and every rule/field.

## Observed consolidated union contract

`comparison.to_record()['union_structure']` uses schema `ml-lab.xml-collection-union-structure@1` and contains:

- `source_paths`: verified compared sources only, alphabetically sorted. Failed/stale/symlinked sources do not contribute.
- `root_partitions`: observed expanded root QNames with their members. Incompatible roots remain separate branches of the consolidated inventory.
- `paths`: deterministic, canonical expanded-QName element paths shared with ML-UW-18. Each path has a parent path, `present_in` source IDs, source/absence counts, exact `occurrences_by_source` and `text_occurrences_by_source`, aggregate counts, and `repeated_in`, `optional_in`, `mixed_content_in` flags.
- `attributes`: canonical expanded attribute QNames nested under the containing element path, with IDs compatible with ML-UW-18 (`attribute:{element_path}@{QName}`), source coverage and exact observed occurrence counts.
- `groups`: group IDs and members with references to the union's `path_ids`; `optional_path_ids` mark paths absent from **at least one** source in that group; shared record-root candidates are preserved. The group views reference the one consolidated path inventory rather than duplicating all path objects.
- `merge_authorized: false` and a `null_policy` marker: these are **structural observations**, not an approved schema, normalization rule, dataset, or permission to combine data.

Counts are derived from the existing safety-checked `XmlStructureArtifact` profiles, not recalculated from values or inferred across absent paths. The union is source-level evidence; **per-record** `path_absent`, `attribute_absent`, `explicit_xsi_nil`, and `empty_element` distinctions come from ML-UW-18's `preview_xml_null_provenance` and will be joined into recipes in ML-UW-20. The source XML is never edited to create a missing element.

The union report is available through the existing XML collection inventory page as expandable JSON. Interactive trees/graphs, row-level null filling, reserved `Comments` output, and CSV/TSV NA round-trip are *not* part of this increment.

## Verification and rollback

Fixtures cover: low-overlap optional paths grouping, C3 acknowledgment, missing values/nil/optionality without fragmenting, mixed-content/repetition and QName conflicts still reported, exact per-source element/text/attribute counts, namespace prefix equivalence, safe exclusion of mutated files, JSON report saving and ML-UW-18 field inventory interoperability. Existing XML C2/C3/C4 and workbench regression tests must also pass.

Apply after ML-UW-18. Reverting ML-UW-19 removes the new comparison behavior and union keys; it does not modify XML sources, C4/C5 artifacts, or the database. After deployment, validate full Flask/PAH navigation on the user's environment if Flask is unavailable in the local test environment.
