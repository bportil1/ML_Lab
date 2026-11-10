# P5-14 — XML hierarchy navigation

The existing read-only XML structure explorer now provides ancestry breadcrumbs,
parent/previous/next sibling controls, keyboard navigation (Up/Down siblings,
Left parent/collapse, Right expand/first child), and visible minimum/maximum
occurrences per parent on repeated branches. The selected path drives the
existing inspector and record-root controls; no separate XML representation,
parser, or extraction-rule engine is introduced.

Filtering now evaluates **all** sibling subtrees rather than stopping when it
encounters the first match. This prevents stale visibility from previous
searches. Clearing the search preserves the selected path's ancestor expansion.

## Scope and limitations

- Display is based on the already sampled/derived structure profiles, not a
  line-by-line DOM of every XML instance; repeated badges show aggregate
  cardinality, not individual instance indexes.
- No modifications to recipe serialization, extraction semantics, datasets,
  XML collection review, or provenance.
- Browser interaction requires the existing ML_Lab UI stack. Source-only
  tests can be run without Flask; real browser acceptance remains useful.
