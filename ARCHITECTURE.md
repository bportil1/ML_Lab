# ML Lab Architecture

ML Lab is a headless machine-learning execution engine. It is designed to work in three modes without changing its core behavior:

1. standalone Python package / CLI;
2. called by PAH through a thin adapter;
3. called by a domain project such as HSQA_DBN.

## Responsibility boundary

ML Lab owns generic operations:

- estimator registries;
- preprocessing pipelines;
- model/candidate selection;
- generic classification evaluation;
- generic clustering evaluation and stability testing;
- model/result serialization.

A caller owns domain semantics. For example, HSQA_DBN should continue to own MI/node feature construction, layer semantics, CWE labels, lineage, cluster interpretation, and visualization. It should pass a feature matrix into ML Lab rather than moving those responsibilities here.

## Task layout

`classification/` and `clustering/` are peers. Clustering is not routed through classification CV semantics.

Future task families should follow the same pattern, for example:

- regression;
- anomaly detection;
- dimensionality reduction;
- feature selection.

## UI policy

There is intentionally no dedicated visualization application in this repository. Results are returned as Python objects and can be serialized as JSON/CSV/joblib artifacts. PAH or another caller can supply its own visualization surface.
