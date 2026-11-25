# Feature Profiling dependency matrix contrast repair

The bounded A2 dependency matrix remains fully interactive and statistical
values, strengths, ordering, limits and tooltips are unchanged. We replace the
transparent heatmap canvas with an **opaque white-based** scientific table,
explicit dark header/value/link colors and consistent link states, regardless
of the embedding PAH/ML_Lab theme. Rows retain their original sticky labels,
keyboard focus, and clickable pairwise drill-down links.

The heatmap still reflects magnitude relative to the visible maximum, not
significance or causality. This is a style-only change to
`src/ml_lab/ui/static/ml_lab_workbench.css`.
