# P5-16 — XML catalog offline acceptance and remaining gates

## Verified locally

`tests/test_xml_catalog_acceptance_p516.py` exercises the public catalog APIs across source inventory, collection recipe, and typed preview snapshots in a project-local SQLite catalog. The tests cover:

- Explicit refresh and indexed cross-section source SHA-256 / inventory reference queries.
- Read-only catalog queries and pagination, without automatic extraction, dataset registration, or index rebuild.
- Project isolation and section-scoped refresh (refreshing one section leaves other sections intact).
- Tampered typed-preview exclusion with an explicit rejection count, without destroying unrelated source and recipe index rows.
- Database transaction rollback on injected SQLite failure, followed by successful refresh after removing the injected fault. SQLite errors are returned as failed sections instead of unhandled server exceptions.
- Catalog snapshot semantics: changing an underlying XML source does **not** silently update or revalidate a previously observed immutable inventory. The index represents verified snapshot content, not the freshness of external source files.

Run offline with `PYTHONPATH=src python -m pytest -q tests/test_xml*`. The test results are environment-specific and do not establish hosted-browser acceptance.

## Important semantics

1. A rebuild is **transactional per section**, not across all sections. All-section refresh may be partially successful.
2. Invalid snapshot exclusion is reported via `invalid_skipped`; it does not imply a failed SQLite rebuild.
3. Catalog results are metadata about previously published snapshots. `stale_until_rebuilt` is not an external-source freshness guarantee; only a new source observation can establish updated XML contents.
4. Catalog errors never authorize modifying source XML, original recipes, registered datasets, or scientific computations.
5. There is no background monitor or automated refreshing.

## Not verified / deferred acceptance

- Hosted Flask request and response behavior (including authentication, CSRF and negative HTTP cases).
- Actual browser controls, keyboard and navigation acceptance on a running ML_Lab instance.
- Representative large collections, concurrency and performance measurement on deployed project storage.
- Current user checkout patch applicability and full unrestricted test suite.
- P5-15's outstanding end-to-end hosted acceptance.

Do not mark P5-15 or P5-16 fully signed off until those checks are performed in the deployed environment. Keep all original extraction and scientific paths available.
