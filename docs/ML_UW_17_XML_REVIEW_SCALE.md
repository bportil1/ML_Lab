# ML-UW-17 — Uncapped XML collection review (internal)

## Implemented

- **XML-C3 and XML-C4** accept any positive, explicitly chosen inventory `max_files`; they no longer clamp the value to 80. XML Data and XML-C1 navigation forward the full selected count without truncating it.
- **XML-C5 and C7** remove their implicit 80-source default rejection, while preserving optional explicit `max_files` budgets and per-file byte, row, and provenance checks. The XML-C6 source-integrity recovery check also accepts collections above 80.
- Source inspection and full structural pairing remain synchronous; selecting a large value may require substantial CPU, time, and memory. XML inventory continues to fail closed on its *explicit* file budget, parsing safety, cumulative size, and entry-count limits. Review and recipe selection do not silently sample or omit sources.
- User-specified source budgets still work, including programmatic limits passed to XML-C5 and C7. No extraction happens automatically simply because a review is opened.

## Future TODO — Pairwise selection sampling

**Priority: future performance/UI enhancement (not implemented in ML-UW-17).** Add an optional, user-controlled sampling mode for large XML pairwise comparison/review with reproducible random seeds and stratified or within/between-group sampling options. Document the population size, sampled pairs, coverage, seed, sample design, and excluded/unevaluated pairs; show explicit *sampled evidence* indicators. Keep an exhaustive mode. Do not treat an unobserved pair as compatible, and never authorize shared recipes or consolidated export on incomplete pairwise evidence without an independently verified full-scope compatibility check. Consider streaming, pagination and background jobs before large-scale interactive comparisons.

## Validation / rollback

Run `PYTHONPATH=src pytest -q tests/test_ml_uw_17_xml_review_uncapped.py tests/test_xml_collection_review.py tests/test_xml_collection_recipe.py tests/test_xml_collection_export.py`. Runtime-mounted browser checks need Flask installed. Reverting this patch restores the old 80-source behavior and associated UI, without changing saved collection output formats or SQLite.
