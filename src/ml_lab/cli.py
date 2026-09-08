from __future__ import annotations

import argparse
import json

from ml_lab import classification, clustering, regression, representation
from ml_lab.classification.reporting import save_results as save_classification_results
from ml_lab.clustering.reporting import save_results as save_clustering_results
from ml_lab.regression.reporting import save_results as save_regression_results
from ml_lab.representation.reporting import save_result as save_representation_result
from ml_lab.registry import list_estimators
from ml_lab.core.serialization import to_jsonable


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ml-lab",
        description="Headless reusable scikit-learn operations engine",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list-estimators", help="List registered estimators")
    listing.add_argument("--task", choices=["all", "classification", "clustering", "regression"], default="all")

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

    regress = sub.add_parser("regress", help="Run regression model selection")
    regress.add_argument("csv", nargs="+", help="CSV file(s) containing features and target column(s)")
    regress.add_argument("--target", nargs="+", default=["target"], help="One or more numeric target columns")
    regress.add_argument("--algorithms", nargs="+", default=["ridge", "random_forest", "extra_trees", "svr", "knn"])
    regress.add_argument("--test-size", type=float, default=0.2)
    regress.add_argument("--cv", type=int, default=5)
    regress.add_argument("--scoring", default="neg_root_mean_squared_error")
    regress.add_argument("--scaling", choices=["auto", "none", "standard", "minmax", "robust"], default="auto")
    regress.add_argument("--n-jobs", type=int, default=-1)
    regress.add_argument("--random-state", type=int, default=42)
    regress.add_argument("--output", default="ml_lab_results/regression")

    cluster = sub.add_parser("cluster", help="Run clustering candidate search/evaluation")
    cluster.add_argument("csv", nargs="+", help="CSV file(s) containing feature columns")
    cluster.add_argument("--label", default=None, help="Optional external-validation label column")
    cluster.add_argument("--algorithms", nargs="+", default=["kmeans", "agglomerative", "spectral", "birch", "gmm"])
    cluster.add_argument("--selection-metric", choices=["silhouette", "calinski_harabasz", "davies_bouldin", "stability"], default="silhouette")
    cluster.add_argument("--repeats", type=int, default=3)
    cluster.add_argument("--scaling", choices=["auto", "none", "standard", "minmax", "robust"], default="auto")
    cluster.add_argument("--random-state", type=int, default=42)
    cluster.add_argument("--output", default="ml_lab_results/clustering")

    represent = sub.add_parser("represent", help="Create a lower-dimensional representation or reconstruction")
    represent.add_argument("csv", nargs="+", help="CSV file(s) containing numeric features")
    represent.add_argument("--exclude", nargs="*", default=[], help="Columns to exclude before representation")
    represent.add_argument("--method", choices=["pca", "mlp_autoencoder", "transformer_autoencoder"], default="pca")
    represent.add_argument("--components", type=int, default=2, help="PCA component count")
    represent.add_argument("--latent-dim", type=int, default=8, help="MLP autoencoder latent dimension")
    represent.add_argument("--hidden-dims", nargs="+", type=int, default=[64, 32])
    represent.add_argument("--activation", choices=["relu", "gelu", "tanh", "leaky_relu"], default="relu")
    represent.add_argument("--dropout", type=float, default=0.0)
    represent.add_argument("--epochs", type=int, default=100)
    represent.add_argument("--batch-size", type=int, default=64)
    represent.add_argument("--learning-rate", type=float, default=1e-3)
    represent.add_argument("--validation-fraction", type=float, default=0.1)
    represent.add_argument("--patience", type=int, default=15)
    represent.add_argument("--scaling", choices=["none", "standard", "minmax", "robust"], default="standard")
    represent.add_argument("--random-state", type=int, default=42)
    represent.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    represent.add_argument("--checkpoint-path", default=None, help="Optional best-model checkpoint path for neural representation runs")
    represent.add_argument("--token-width", type=int, default=1, help="Adjacent feature count per Transformer token for 2-D input")
    represent.add_argument("--model-dim", type=int, default=64, help="Transformer embedding dimension")
    represent.add_argument("--nhead", type=int, default=4, help="Transformer attention heads")
    represent.add_argument("--encoder-layers", type=int, default=2)
    represent.add_argument("--decoder-layers", type=int, default=2)
    represent.add_argument("--feedforward-dim", type=int, default=128)
    represent.add_argument("--transformer-activation", choices=["relu", "gelu"], default="gelu")
    represent.add_argument("--max-tokens", type=int, default=1024)
    represent.add_argument("--output-activation", choices=["none", "sigmoid", "tanh"], default="none")
    represent.add_argument("--output", default="ml_lab_results/representation")

    experimental = sub.add_parser("experimental", help="Inspect or run isolated experimental modules")
    experimental_sub = experimental.add_subparsers(dest="experimental_command", required=True)
    experimental_list = experimental_sub.add_parser("list", help="List experimental module manifests without importing them")
    experimental_list.add_argument("--status", default=None)
    experimental_list.add_argument("--capability", default=None)
    experimental_info = experimental_sub.add_parser("info", help="Show one experimental module manifest")
    experimental_info.add_argument("experiment_id")
    experimental_run = experimental_sub.add_parser("run", help="Run an explicitly selected experimental entrypoint")
    experimental_run.add_argument("experiment_id")
    experimental_run.add_argument("--kwargs-json", default="{}", help="JSON object passed as keyword arguments")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "list-estimators":
        for spec in list_estimators(args.task):
            print(f"{spec.task:14s} {spec.id:20s} {spec.name}")
        return 0

    if args.command == "experimental":
        from ml_lab.experimental import get_manifest, list_experiments, run_experiment

        if args.experimental_command == "list":
            for manifest in list_experiments(status=args.status, capability=args.capability):
                capabilities = ",".join(manifest.capabilities) or "-"
                print(f"{manifest.id:28s} {manifest.status:12s} {capabilities:28s} {manifest.name}")
            return 0
        if args.experimental_command == "info":
            print(json.dumps(to_jsonable(get_manifest(args.experiment_id).to_record()), indent=2, sort_keys=True))
            return 0
        kwargs = json.loads(args.kwargs_json)
        if not isinstance(kwargs, dict):
            raise SystemExit("--kwargs-json must decode to a JSON object")
        result = run_experiment(args.experiment_id, **kwargs)
        print(json.dumps(to_jsonable(result), indent=2, sort_keys=True))
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

    if args.command == "represent":
        X = representation.load_csv_features(args.csv, exclude_columns=args.exclude)
        if args.method == "pca":
            result = representation.pca(
                X,
                config=representation.PCARepresentationConfig(
                    n_components=args.components,
                    scaling=args.scaling,
                    random_state=args.random_state,
                ),
            )
        elif args.method == "mlp_autoencoder":
            result = representation.autoencode(
                X,
                model_config=representation.MLPAutoencoderConfig(
                    hidden_dims=tuple(args.hidden_dims),
                    latent_dim=args.latent_dim,
                    activation=args.activation,
                    output_activation=args.output_activation,
                    dropout=args.dropout,
                ),
                training_config=representation.AutoencoderTrainingConfig(
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    validation_fraction=args.validation_fraction,
                    patience=args.patience,
                    scaling=args.scaling,
                    random_state=args.random_state,
                    device=args.device,
                    checkpoint_path=args.checkpoint_path,
                ),
            )
        else:
            result = representation.transformer_autoencode(
                X,
                model_config=representation.TransformerAutoencoderConfig(
                    token_width=args.token_width,
                    model_dim=args.model_dim,
                    latent_dim=args.latent_dim,
                    nhead=args.nhead,
                    encoder_layers=args.encoder_layers,
                    decoder_layers=args.decoder_layers,
                    feedforward_dim=args.feedforward_dim,
                    dropout=args.dropout,
                    transformer_activation=args.transformer_activation,
                    output_activation=args.output_activation,
                    max_tokens=args.max_tokens,
                ),
                training_config=representation.AutoencoderTrainingConfig(
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    learning_rate=args.learning_rate,
                    validation_fraction=args.validation_fraction,
                    patience=args.patience,
                    scaling=args.scaling,
                    random_state=args.random_state,
                    device=args.device,
                    checkpoint_path=args.checkpoint_path,
                ),
            )
        output = save_representation_result(result, args.output)
        print(f"Results written to: {output}")
        print(
            f"method={result.method} latent_shape={tuple(result.latent.shape)} "
            f"RMSE={result.metrics['rmse']:.6f} MAE={result.metrics['mae']:.6f}"
        )
        return 0

    if args.command == "regress":
        X, y = regression.load_csv_dataset(args.csv, target_columns=args.target)
        config = regression.RegressionSearchConfig(
            cv_folds=args.cv,
            scoring=args.scoring,
            n_jobs=args.n_jobs,
            random_state=args.random_state,
            scaling=args.scaling,
        )
        results = regression.run(
            X,
            y,
            estimators=args.algorithms,
            config=config,
            test_size=args.test_size,
            target_names=args.target,
        )
        output = save_regression_results(results, args.output)
        print(f"Results written to: {output}")
        for rank, result in enumerate(results, start=1):
            print(
                f"{rank:2d}. {result.estimator_id:20s} "
                f"CV={result.best_cv_score:.4f} RMSE={result.metrics['rmse']:.4f} "
                f"R2={result.metrics['r2']:.4f}"
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
