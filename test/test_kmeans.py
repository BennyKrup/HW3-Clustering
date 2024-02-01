# Write your k-means unit tests here
import pytest
from cluster.kmeans import KMeans
import numpy as np

def test_kmeans_initialization():
    """
    Test KMeans class initialization.
    """
    k = 3
    tol = 1e-4
    max_iter = 100
    kmeans = KMeans(k=k, tol=tol, max_iter=max_iter)
    assert kmeans.num_clusters == k
    assert kmeans.tolerance == tol
    assert kmeans.max_iterations == max_iter

def test_kmeans_fit():
    """
    Test the fit method of KMeans class.
    """
    kmeans = KMeans(k=3, tol=1e-4, max_iter=100)
    data = np.array([[1, 2], [1, 4], [1, 0],
                     [10, 2], [10, 4], [10, 0]])
    kmeans.fit(data)
    assert kmeans.centroids.shape == (3, 2)
    assert kmeans.iteration <= kmeans.max_iterations

def test_kmeans_predict():
    """
    Test the predict method of KMeans class.
    """
    kmeans = KMeans(k=3, tol=1e-4, max_iter=100)
    data = np.array([[1, 2], [1, 4], [1, 0],
                     [10, 2], [10, 4], [10, 0]])
    kmeans.fit(data)
    predictions = kmeans.predict(data)
    assert len(predictions) == len(data)
    assert set(predictions) <= set(range(3))  # predictions should be among the cluster indices

def test_kmeans_edge_cases():
    """
    Test KMeans behavior with edge cases like one cluster or more clusters than points.
    """
    data = np.array([[1, 2], [1, 4], [1, 0]])

    # More clusters than points
    kmeans = KMeans(k=4, tol=1e-4, max_iter=100)
    with pytest.raises(ValueError):
        kmeans.fit(data)

    # One cluster
    kmeans = KMeans(k=1, tol=1e-4, max_iter=100)
    kmeans.fit(data)
    assert kmeans.centroids.shape == (1, 2)
