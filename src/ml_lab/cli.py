from __future__ import annotations

import argparse

from ml_lab import classification, clustering
from ml_lab.classification.reporting import save_results as save_classification_results
from ml_lab.clustering.reporting import save_results as save_clustering_results
from ml_lab.registry import list_estimators


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-lab",
        description="Headless reusable scikit-learn operations engine",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list-estimators", help="List registered estimators")
    listing.add_argument("--task", choices=["all", "classification", "clustering"], default="all")

    classify = sub.add_parser("classify", help="Run classification model selection")
    classify.add_argument("csv", nargs="+", help="CSV file(s) containing features and a label column")
    classify.add_argument("--label", default="label")
    classify.add_argument("--algorithms", nargs="+", default=["rf", "hgbc", "lda", "qda", "ridge"])
    classify.add_argument("--test-size", type=float, default=0.2)
    classify.add_argument("--cv", type=int, default=5)
    classify.add_argument("--scoring", default="accuracy")
    classify.add_argument("--scaling", choices=["auto", "none", "standard", "minmax", "robust"], default="auto")
    classify.add_argument("--n-jobs", type=int, default=-1)
    classify.add_argument("--random-state", type=int, default=42)
    classify.add_argument("--output", default="ml_lab_results/classification")

    cluster = sub.add_parser("cluster", help="Run clustering candidate search/evaluation")
    cluster.add_argument("csv", nargs="+", help="CSV file(s) containing feature columns")
    cluster.add_argument("--label", default=None, help="Optional external-validation label column")
    cluster.add_argument("--algorithms", nargs="+", default=["kmeans", "agglomerative", "spectral", "birch", "gmm"])
    cluster.add_argument("--selection-metric", choices=["silhouette", "calinski_harabasz", "davies_bouldin", "stability"], default="silhouette")
    cluster.add_argument("--repeats", type=int, default=3)
    cluster.add_argument("--scaling", choices=["auto", "none", "standard", "minmax", "robust"], default="auto")
    cluster.add_argument("--random-state", type=int, default=42)
    cluster.add_argument("--output", default="ml_lab_results/clustering")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list-estimators":
        for spec in list_estimators(args.task):
            print(f"{spec.task:14s} {spec.id:20s} {spec.name}")
        return 0

    if args.command == "classify":
        X, y = classification.load_csv_dataset(args.csv, label_column=args.label)
        config = classification.ClassificationSearchConfig(
            cv_folds=args.cv,
            scoring=args.scoring,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            scaling=args.scaling,
        )
        results = classification.run(
            X,
            y,
            estimators=args.algorithms,
            config=config,
            test_size=args.test_size,
            label_name=args.label,
        )
        output = save_classification_results(results, args.output)
        print(f"Results written to: {output}")
        for rank, result in enumerate(results, start=1):
            print(
                f"{rank:2d}. {result.estimator_id:16s} "
                f"CV={result.best_cv_score:.4f} accuracy={result.metrics['accuracy']:.4f}"
            )
        return 0

    X, y_true = clustering.load_csv_dataset(args.csv, label_column=args.label)
    config = clustering.ClusteringSearchConfig(
        selection_metric=args.selection_metric,
        repeats=args.repeats,
        random_state=args.random_state,
        scaling=args.scaling,
    )
    results = clustering.run(X, estimators=args.algorithms, config=config, y_true=y_true)
    output = save_clustering_results(results, args.output)
    print(f"Results written to: {output}")
    for rank, result in enumerate(results, start=1):
        display_score = result.metrics.get(args.selection_metric) if args.selection_metric != "stability" else result.stability
        print(f"{rank:2d}. {result.estimator_id:20s} {args.selection_metric}={display_score}")
    return 0
