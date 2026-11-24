# ML_Lab pyPIQUE-style sidebar and dataset readiness (2026-11-24)

## Scope

A **presentation-only** change to ML_Lab's existing five-section workbench. It borrows pyPIQUE's light blue-gray workflow rail, card selection, small status LEDs, two-line card text, typographic hierarchy and bounded details drawer. It does **not** import pyPIQUE functionality, change ML_Lab route or workspace semantics, add a sixth primary tab, replace any active tab's nested links, add database tables, or run scientific operations automatically.

The five top-level sections remain Data, Feature Profiling, Representation, Comparison, and General ML Tools. When **Data** is active, its three preserved subpanels appear *below* all five main entries: Tabular Data, XML Data, and Post-Collection Utilities. Other sections retain their existing contextual links and legacy scientific-tool links. At narrow screen widths, card and link groups reflow without losing their hierarchy.

## Dataset-bound status contract

The rail's **Active dataset** card and each section's one-line status are supplied by the existing `GET /api/workbench/preflight` response, scoped to the current workbench instance. The endpoint reads and verifies project-local saved work; it does **not** select an artifact, train a model, profile a dataset, or write a generated output. The existing header context script continues to validate the selected source separately.

| Preflight condition | Data | Feature Profiling | Representation | Comparison | General ML Tools |
| --- | --- | --- | --- | --- | --- |
| No active dataset, no reusable saved artifact | Ready to create/select dataset | Dataset required | Dataset required | Dataset required | Data-linked tools require selection; independent tools remain available (amber partial status) |
| Saved artifact exists but is not active | Saved data available; explicitly select it | Dataset required | Dataset required | Dataset required | Data-linked tools need selection |
| Verified active dataset | Selected source basename | Ready to run profile; cached profile shown as complete if source-matched | Ready to choose representation | Ready to choose comparison | Ready to configure tools |
| Stale source | Reselect changed dataset | Reselect | Reselect | Reselect | Reselect for data-linked tools |
| Preflight request unavailable | Unknown status, not an inferred failure or success | Unknown | Unknown | Unknown | Unknown |

The top card shows a source basename and verified row/column counts for the active selection. Long names truncate visually while keeping a full-path tooltip. Section statuses are intentionally **one line each**, with accessible full text and optional tooltips. Colored dots are accompanied by explicit textual availability states; color is never the sole information channel.

## Acceptance and boundaries

- HTML/Jinja rendering and static navigation regression tests verify five anchors, active selection, Data subpanels, other sections' contextual links, and the existing scientific-tools section.
- Node.js test replays the actual browser status controller with an isolated, synthetic DOM for six cases: empty, reusable, active, cached profile, stale, and HTTP failure. No external resources are used.
- Existing preflight and context routes are untouched, including project-local DB ownership, source verification, session-generation checks, and no automatic training or recalculation.
- No change to the underlying ML-UW-27–38 scientific API, adapter, model, or results format.
- Flask/PAH deployment/browser acceptance remains a separate required check in the actual user's installed environment. The development environment may skip Flask-dependent tests if Flask is absent.
