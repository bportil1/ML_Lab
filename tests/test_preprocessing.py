import numpy as np

from ml_lab import preprocessing


def test_scale_impute_and_pca_are_simple_public_operations():
    X = np.array([[1.0, 2.0], [2.0, np.nan], [3.0, 6.0], [4.0, 8.0]])
    filled = preprocessing.impute(X, strategy="median")
    assert not np.isnan(filled.data).any()

    scaled = preprocessing.scale(filled.data, "standard")
    assert scaled.data.shape == X.shape
    assert np.allclose(scaled.data.mean(axis=0), 0.0)

    reduced = preprocessing.pca(filled.data, n_components=1)
    assert reduced.data.shape == (4, 1)
    assert reduced.explained_variance_ratio.shape == (1,)
