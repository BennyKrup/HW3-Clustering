# write your silhouette score unit tests here
import numpy as np
import pytest
from cluster.silhouette import Silhouette
from cluster.utils import make_clusters
from sklearn.metrics import silhouette_samples
from sklearn.datasets import make_blobs
from cluster.kmeans import KMeans


def test_kmeans_silhouette_integration():
    # Generate a simple dataset
    mat, truth = make_clusters(n=500, m=2, k=3, scale=1)

    # Run KMeans
    kmeans = KMeans(k=3, tol=1e-4, max_iter=300)
    kmeans.fit(mat)
    pred = kmeans.predict(mat)

    # Calculate silhouette scores using the custom Silhouette class
    silhouette = Silhouette()
    custom_scores = silhouette.score(mat, pred)

    # Calculate silhouette scores using sklearn
    sklearn_scores = silhouette_samples(mat, pred)

    # The scores should be relatively similar (range of -1 to 1)
    assert np.allclose(custom_scores, sklearn_scores, atol=1e-1), "Custom silhouette scores do not match sklearn's scores"


def test_silhouette_single_cluster():
    """
    Test silhouette scores for a single cluster scenario.
    In this case, the silhouette score is not defined and should raise a ValueError.
    """
    X = np.array([[1, 2], [2, 3], [3, 4]])  # All points in a single cluster
    y = np.array([0, 0, 0])  # Only one cluster label
    silhouette = Silhouette()
    
    # Expect ValueError because silhouette scores can't be calculated for a single cluster
    with pytest.raises(ValueError):
        silhouette.score(X, y)

def test_silhouette_more_clusters_than_points():
    """
    Test silhouette scores for the scenario where there are more clusters than points.
    This test assumes that an error should be raised as it's an invalid scenario for silhouette scores.
    """
    X = np.array([[1, 2], [2, 3]])  # Only two points
    y = np.array([0, 1])  # Two different cluster labels, which is okay
    y_invalid = np.array([0, 1, 2])  # Invalid cluster labels
    silhouette = Silhouette()
    
    # Valid scenario: two points in two clusters
    valid_scores = silhouette.score(X, y)
    assert len(valid_scores) == 2, "Silhouette scores should be calculated for two points"
    
    # Invalid scenario: more clusters than points should raise ValueError
    with pytest.raises(ValueError):
        silhouette.score(X, y_invalid)