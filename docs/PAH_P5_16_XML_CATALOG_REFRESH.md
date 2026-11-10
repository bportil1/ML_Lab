# P5-16 — Explicit XML catalog index refresh

The XML catalog browser gains an explicit POST action to rebuild one or all project-local index sections from **existing immutable** inventory, recipe, and typed-artifact snapshots. It does not rescan XML sources, execute XML extraction, register datasets, or alter snapshots. Browsing remains read-only.

An all-section refresh is **not** an all-or-nothing transaction: each section is individually transactional; if one fails, other sections may have refreshed. The report names successes, errors, counts, and rejected artifacts. A rejected snapshot is omitted from the refreshed section and counted; this does not count as a fatal rebuild failure. The SQLite index remains disposable and potentially stale until rebuilt. Sources and artifacts remain authoritative. A foreign Origin is rejected for POST when supplied, and an explicit confirmation field is required; deployments must still apply their normal authentication and CSRF controls.

This increment does not verify Flask/browser behavior. It retains the P5-15 hosted acceptance gate and P5-16 full integration gate.
