from __future__ import annotations

from typing import Any, Callable

from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import DotProduct, Matern, RBF, RationalQuadratic
from sklearn.linear_model import PassiveAggressiveClassifier, RidgeClassifier, SGDClassifier
from sklearn.naive_bayes import BernoulliNB, ComplementNB, GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import LinearSVC, NuSVC, SVC
from sklearn.tree import DecisionTreeClassifier

from ml_lab.core.specs import EstimatorSpec


def _rf(seed: int) -> BaseEstimator:
    return RandomForestClassifier(random_state=seed, n_jobs=1)


def _etc(seed: int) -> BaseEstimator:
    return ExtraTreesClassifier(random_state=seed, n_jobs=1)


def _knn(_: int) -> BaseEstimator:
    return KNeighborsClassifier(n_jobs=1)


def _svc(seed: int) -> BaseEstimator:
    return SVC(probability=True, random_state=seed)


def _nusvc(seed: int) -> BaseEstimator:
    return NuSVC(probability=True, random_state=seed)


def _gp(seed: int) -> BaseEstimator:
    return GaussianProcessClassifier(random_state=seed)


def _mlp(seed: int) -> BaseEstimator:
    return MLPClassifier(random_state=seed, max_iter=1000)


def _sgd(seed: int) -> BaseEstimator:
    return SGDClassifier(random_state=seed)


def _pa(seed: int) -> BaseEstimator:
    return PassiveAggressiveClassifier(random_state=seed)


def _dt(seed: int) -> BaseEstimator:
    return DecisionTreeClassifier(random_state=seed)


def _hgbc(seed: int) -> BaseEstimator:
    return HistGradientBoostingClassifier(random_state=seed)


def _ada(seed: int) -> BaseEstimator:
    return AdaBoostClassifier(random_state=seed)


def _lsvc(seed: int) -> BaseEstimator:
    return LinearSVC(random_state=seed)


def _const(factory: Callable[[], BaseEstimator]) -> Callable[[int], BaseEstimator]:
    return lambda _seed: factory()


# The search spaces intentionally preserve the spirit of the original repository,
# but are represented declaratively so they can be inspected/extended without
# adding another hand-written optimization loop.
_REGISTRY: dict[str, EstimatorSpec] = {
    "knn": EstimatorSpec(
        "knn", "K-Nearest Neighbors", "classification", _knn,
        [{
            "n_neighbors": list(range(1, 9)),
            "weights": ["uniform", "distance"],
            "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
            "metric": ["euclidean", "manhattan", "chebyshev", "canberra"],
        }],
        preprocess="standard",
        notes="Mahalanobis/custom inverse-weight legacy branches are intentionally not implicit; add them as explicit specs if required.",
    ),
    "svc": EstimatorSpec(
        "svc", "Support Vector Classifier", "classification", _svc,
        [
            {"kernel": ["linear"], "C": [0.1, 1.0, 10.0], "tol": [1e-4, 1e-3, 1e-2]},
            {"kernel": ["rbf"], "C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"], "tol": [1e-4, 1e-3, 1e-2]},
            {"kernel": ["sigmoid"], "C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"], "coef0": [0.0, 0.5, 1.0], "tol": [1e-4, 1e-3, 1e-2]},
            {"kernel": ["poly"], "degree": [1, 2, 3, 5], "C": [0.1, 1.0, 10.0], "gamma": ["scale", "auto"], "coef0": [0.0, 0.5, 1.0], "tol": [1e-4, 1e-3, 1e-2]},
        ],
        preprocess="standard",
    ),
    "nsvc": EstimatorSpec("nsvc", "Nu-SVC", "classification", _nusvc, [{"nu": [0.25, 0.5, 0.75], "kernel": ["rbf", "linear"]}], preprocess="standard"),
    "gp": EstimatorSpec(
        "gp", "Gaussian Process Classifier", "classification", _gp,
        [{"kernel": [RBF(0.5), RBF(1.0), Matern(length_scale=1.0, nu=1.5), RationalQuadratic(), DotProduct()]}],
        preprocess="standard",
    ),
    "dt": EstimatorSpec("dt", "Decision Tree", "classification", _dt, [{"criterion": ["gini", "entropy", "log_loss"], "max_depth": [None, 3, 5, 10]}]),
    "rf": EstimatorSpec(
        "rf", "Random Forest", "classification", _rf,
        [{"n_estimators": [50, 100, 150], "criterion": ["gini", "entropy", "log_loss"], "max_features": ["sqrt", "log2", None], "min_samples_split": [2, 3, 5, 6], "min_samples_leaf": [1, 2, 3, 4, 5]}],
    ),
    "hgbc": EstimatorSpec(
        "hgbc", "Histogram Gradient Boosting", "classification", _hgbc,
        [{"max_iter": [100, 125, 135], "learning_rate": [0.1, 0.2, 0.3], "l2_regularization": [0.0, 1.0, 2.0]}],
    ),
    "ada": EstimatorSpec("ada", "AdaBoost", "classification", _ada, [{"n_estimators": [50, 100, 125, 135], "learning_rate": [0.1, 0.2, 0.3, 1.0]}]),
    "qda": EstimatorSpec("qda", "Quadratic Discriminant Analysis", "classification", _const(QuadraticDiscriminantAnalysis), [{"reg_param": [0.0, 0.5, 1.0]}], preprocess="standard"),
    "lda": EstimatorSpec(
        "lda", "Linear Discriminant Analysis", "classification", _const(LinearDiscriminantAnalysis),
        [
            {"solver": ["svd"]},
            {"solver": ["lsqr", "eigen"], "shrinkage": [None, "auto"]},
        ],
        preprocess="standard",
    ),
    "mlp": EstimatorSpec(
        "mlp", "Multilayer Perceptron", "classification", _mlp,
        [{"activation": ["identity", "logistic", "tanh", "relu"], "hidden_layer_sizes": [(100,), (125,)], "solver": ["adam"]}],
        preprocess="standard",
    ),
    "ridge": EstimatorSpec(
        "ridge", "Ridge Classifier", "classification", _const(RidgeClassifier),
        [{"alpha": [0.5, 1.0, 1.5], "fit_intercept": [True, False], "solver": ["auto", "svd", "cholesky", "lsqr", "sparse_cg", "sag", "saga"]}],
        preprocess="standard",
    ),
    "pa": EstimatorSpec(
        "pa", "Passive Aggressive", "classification", _pa,
        [{"loss": ["hinge", "squared_hinge"], "C": [0.1, 1.0, 10.0], "tol": [1e-4, 1e-3, 1e-2]}],
        preprocess="standard",
    ),
    "sgd": EstimatorSpec(
        "sgd", "SGD Classifier", "classification", _sgd,
        [{
            "loss": ["hinge", "log_loss", "modified_huber", "squared_hinge", "perceptron"],
            "penalty": ["l2", "l1", "elasticnet", None],
            "alpha": [1e-4, 1e-3],
            "tol": [1e-4, 1e-3, 1e-2],
        }],
        preprocess="standard",
    ),
    "etc": EstimatorSpec(
        "etc", "Extra Trees", "classification", _etc,
        [{"n_estimators": [50, 100, 150], "criterion": ["gini", "entropy", "log_loss"], "max_features": ["sqrt", "log2", None], "min_samples_split": [2, 3, 5, 6], "min_samples_leaf": [1, 2, 3, 4, 5]}],
    ),
    "gnb": EstimatorSpec("gnb", "Gaussian Naive Bayes", "classification", _const(GaussianNB), [{"var_smoothing": [1e-7, 1e-8, 1e-9, 1e-10]}]),
    "mnb": EstimatorSpec("mnb", "Multinomial Naive Bayes", "classification", _const(MultinomialNB), [{"alpha": [0.0, 0.8, 0.9, 1.0, 1.1], "force_alpha": [True, False]}], preprocess="minmax"),
    "compnb": EstimatorSpec("compnb", "Complement Naive Bayes", "classification", _const(ComplementNB), [{"alpha": [0.0, 0.8, 0.9, 1.0, 1.1], "force_alpha": [True, False]}], preprocess="minmax"),
    "bnb": EstimatorSpec("bnb", "Bernoulli Naive Bayes", "classification", _const(BernoulliNB), [{"alpha": [0.0, 0.8, 0.9, 1.0, 1.1], "force_alpha": [True, False]}], preprocess="minmax"),
    "lsvc": EstimatorSpec(
        "lsvc", "Linear SVC", "classification", _lsvc,
        [
            {"penalty": ["l2"], "loss": ["hinge", "squared_hinge"], "C": [0.5, 0.7, 1.0, 1.3, 1.5], "dual": [True]},
            {"penalty": ["l1"], "loss": ["squared_hinge"], "C": [0.5, 0.7, 1.0, 1.3, 1.5], "dual": [False]},
        ],
        preprocess="standard",
    ),
}


def list_estimators() -> list[EstimatorSpec]:
    return [_REGISTRY[key] for key in sorted(_REGISTRY)]


def get_estimator_spec(estimator_id: str) -> EstimatorSpec:
    try:
        return _REGISTRY[estimator_id]
    except KeyError as exc:
        supported = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"unknown estimator {estimator_id!r}; supported: {supported}") from exc


def prefixed_param_grid(spec: EstimatorSpec) -> list[dict[str, list[Any]]]:
    return [{f"estimator__{key}": value for key, value in grid.items()} for grid in spec.param_grid]
