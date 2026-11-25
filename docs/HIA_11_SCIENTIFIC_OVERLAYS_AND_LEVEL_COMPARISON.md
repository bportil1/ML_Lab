# HIA-11 — Scientific graph overlays and matched-cohort comparisons

## Ownership and scope

- ML_Lab owns the empirical statistics and the opt-in HIA-11 presentation controller.
- PAH_Shared continues to own the actual graph SVG, graph navigation, selection,
  node/edge identities, grouping, and pagination. No second graph engine is added.
- Original graph payload, observations, HIA-04/HIA-07/HIA-10 report objects,
  persistent storage, and topology are unmodified. Only SVG *presentation*
  properties on currently rendered node elements change.

## Statistical provenance and meaning

HIA-11 consumes reports **only after a user explicitly computes them** and after
checking both the current source revision and the validated source digest.
An edited mapping or estimator setting invalidates its published report and
immediately withdraws its related colors.

Three distinct overlay sources are exposed:

1. **HIA-04 atomic**: measured entropy or observed scalar-row support for exactly
   one explicitly bound source node.
2. **HIA-07 level**: entropy, conditional remainder given mapped peers,
   MI with other mapped peers, or support for each explicitly mapped original
   node on the selected level. These are empirical same-cohort estimates.
3. **HIA-10 network**: measured H(level), H(level | earlier mapped levels),
   I(level; earlier), within-level TC, incremental target information, or
   matched-cohort support. These are **level aggregates**. They are repeated
   visually across that level's explicitly mapped nodes; they do not imply a
   measured atomic value or a learned probabilistic edge.

Unmapped or unmeasured source nodes are neutral, never treated as zero. A
measured zero stays a measured zero. Colors are relative to the active overlay
(max within the selected mapped set) and not comparable across unlike reports
or separate cohorts.

## Level comparison

Two structural levels are compared only within **one HIA-10 matched complete-case
cohort**, without combining results from independent HIA-07 reports. The
comparison includes per-level entropy, conditional entropy, information
shared with preceding levels, within-level total correlation, mapped coverage,
and target-information increment when a target is declared. Pairwise
interlevel MI is reported separately; it is **not additive** and not a causal
interpretation of the structural edges.

## Exports and scale

- CSV exports all explicitly mapped values for the active scientific overlay,
  including exact original node IDs, metric, scope, source digest, and support;
  it does not sample or truncate those mapped rows.
- SVG and PNG are **the currently visible rendered graph page**, not a claim to
  export all original nodes on a large graph. The SVG includes inline colors
  and a light background for readability when opened separately.
- Existing HIA-03 complete graph JSON and HIA-10 complete scientific report JSON
  remain the authoritative full-source exports.
- Overlay painting touches the bounded set of SVG nodes currently drawn by
  PAH_Shared, even when the original graph contains far more nodes.
- No new network services, database writes, or external asset dependencies.

## Acceptance

`node --test tests/test_hia11_scientific_visuals.mjs` checks mathematical scope,
zero-vs-missing semantics, same-cohort comparisons, original IDs, and exports.
`python -m pytest -q tests/test_hia11_visualization_contract.py` checks
owner/UI asset loading, source-revision validation and existing viewer usage.

Live-browser checks to complete on the user's machine: confirm HIA-04/HIA-07/
HIA-10 reports appear in overlay mode, verify old overlay clears on remapping,
inspect overview and paged-node navigation, verify selection/fullscreen/popout,
check PNG/SVG readability and CSV row coverage, and test low and high contrast.
