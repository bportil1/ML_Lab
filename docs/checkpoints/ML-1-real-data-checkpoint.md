# ML-1 Real-Data Checkpoint

## Scope

This checkpoint exercised ML_Lab 0.16.0 (A1 intake/inventory, A2 statistical profiling, and A2.1 source/profile interaction contracts) against nine real CSV files already available in the local project file set. All files were profiled using the complete valid-row population (`max_rows=0`); no statistical sampling was used. No source files were modified.

## Corpus results

| File | Rows | Columns | Duplicate rows | All-missing cols | Constant cols | Missingness signals | IQR-outlier signals | Relationships |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Data.csv | 15 | 31 | 0 | 1 | 25 | 2 | 0 | 10 |
| dependency(1).csv | 756 | 14 | 0 | 1 | 5 | 0 | 1 | 28 |
| dependency.csv | 84 | 11 | 0 | 0 | 2 | 0 | 4 | 36 |
| energy(1).csv | 1080 | 14 | 0 | 1 | 4 | 0 | 0 | 36 |
| energy.csv | 120 | 11 | 0 | 0 | 1 | 0 | 3 | 45 |
| other.csv | 32 | 11 | 0 | 0 | 2 | 4 | 5 | 36 |
| paper_report.csv | 445 | 4 | 0 | 0 | 0 | 0 | 0 | 0 |
| reconstruction(1).csv | 540 | 14 | 0 | 1 | 4 | 0 | 1 | 36 |
| reconstruction.csv | 60 | 11 | 0 | 0 | 1 | 0 | 4 | 45 |

All nine files parsed successfully. No malformed-width rows or duplicate rows were observed in this checkpoint corpus. The profiler emitted **74 quality signals**: 4 error-level all-missing-column signals, 46 warnings, and 24 informational signals. The dominant patterns were 44 constant columns, 18 IQR-outlier signals, 6 moderate-missingness signals, 4 all-missing columns, and 2 class-imbalance warnings.

## What worked as intended

- Recursive intake and CSV parsing worked across all nine files without parse failures.
- Full-population profiling completed on each dataset.
- Constant metadata columns and completely empty columns were surfaced clearly.
- Missingness, IQR outliers, and categorical imbalance were detected without changing the source.
- Numeric relationship profiling produced bounded Pearson/Spearman/discretized-MI outputs where eligible.
- The A2.1 table contract was exercised on the 445-row paper report: default 50-row paging returned rows 1–50 while preserving access to all 445 rows; global search for `causal` matched five rows; sorting worked across the full source.
- Profile persistence produced a reusable run artifact containing all nine dataset profiles.

## Important real-data findings / profiler blind spots

1. **0/1 numeric semantics can be misclassified as boolean.** In `Data.csv`, fields such as `generation_condition_index`, `dataset_replica`, `dbn_depth`, and `layer` are inferred as boolean because their observed values are only 0/1. Some are semantically indices/counts rather than truth values. A3 therefore needs explicit type overrides/coercion, and the profiling UI should make inferred-vs-user-declared type visible.
2. **High-cardinality path/name columns are not always marked as identifiers.** In `paper_report.csv`, `Path` is unique for all 445 rows but is treated as text, not identifier-like. This is not necessarily wrong, but it shows that identifier detection cannot be relied on as an automatic deletion rule. A3 should offer suggestions, never silently drop identifiers.
3. **Constant/all-missing columns are common in experiment exports.** The HSQA result tables contain repeated metadata columns and, in several files, an entirely empty `aug_data_path`. A3 should support explicit removal of constant/all-missing columns with preview and reversible recipe entries.
4. **Missing values are structural in summary/export tables.** Missing `value`/summary-statistic cells occur because some statistics are undefined rather than because the file is corrupt. A3 must distinguish actions such as keep-as-missing, drop, fill, or sentinel conversion; automatic imputation would be inappropriate.
5. **Outlier flags are contextual, not automatic errors.** Several energy/dependency metrics naturally trigger IQR outlier rules. A3 should allow filtering/winsorization only as explicit user-selected transformations, never treat outlier detection as a cleaning mandate.
6. **Experiment metadata mixes categorical and numeric-looking values.** Fields such as augmentation ratio are represented as labels (`aug_ratio_0`, etc.). A3 should support parsing/derived-column operations so users can preserve the original label while deriving a numeric augmentation field when useful.
7. **Path/string normalization is a real need.** The paper-report dataset contains long filesystem paths and filenames. A3 should support string trim/replace/case operations and path-derived columns without overwriting originals.

## ML-2 requirements established by this checkpoint

The controlled-transformation sprint should prioritize: explicit type override/coercion; missing/sentinel handling; drop/select/rename/reorder columns; optional removal of all-missing and constant columns; row filters; duplicate removal; string/path cleanup; derived columns (including parsing numeric values from labels); normalization/standardization; categorical encoding; preview-before-apply; and persistent transformation recipes that record row/column/value changes and coercion failures. Outlier treatment must remain opt-in. Raw sources must never be overwritten by default.

## Checkpoint status

**ML-1 passes as a real-data checkpoint for the currently available corpus.** It exposed concrete transformation requirements without requiring additional Data Lab architecture. If a separate larger/messier dataset collection is intended as the canonical corpus, run the same checkpoint against that directory before freezing ML-2 edge-case behavior.
