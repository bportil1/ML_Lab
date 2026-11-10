# P5-16 — Read-only XML catalog workspace

- `/data/xml/catalog` displays one bounded, paginated index query at a time. Tabs: collections, sources, recipes, typed artifacts.
- The selected project root comes from the mounted ML_Lab blueprint, not URL parameters.
- Source SHA-256, inventory ID, kind, and page size filters are validated by the producer-owned query APIs.
- The UI does not automatically scan/rebuild the index, inspect source files, publish snapshots, or authorize extraction. Indexed metadata may be stale.
- Links lead to existing collection/recipe/Data Lab workflows, without trusting stored absolute paths as actionable navigation targets.
- Production Flask/browser behavior remains a separate acceptance gate.
