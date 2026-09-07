from sklearn.datasets import load_breast_cancer, load_iris

from ml_lab import classification, clustering

# Classification
cancer = load_breast_cancer(as_frame=True)
classification_results = classification.run(
    cancer.data,
    cancer.target,
    estimators=["rf", "lda"],
    config=classification.ClassificationSearchConfig(cv_folds=3, n_jobs=1),
)
print(classification_results[0].to_record())

# Clustering; target is optional and is evaluation-only here.
iris = load_iris(as_frame=True)
cluster_results = clustering.run(
    iris.data,
    estimators=["kmeans", "agglomerative"],
    config=clustering.ClusteringSearchConfig(repeats=2),
    y_true=iris.target,
)
print(cluster_results[0].to_record())
