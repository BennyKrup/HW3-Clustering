import numpy as np
from scipy.spatial.distance import cdist, pdist, squareform


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """
        pass

    def _validate_input(self, X, y):
        if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
            raise ValueError("X and y must be numpy arrays")
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        if len(X) != len(y):
            raise ValueError("X and y must have the same number of rows")
        if len(np.unique(y)) < 2:
            raise ValueError("y must have at least two unique values")

        


    def score(self, X: np.ndarray, y: np.ndarray,vectorized = False) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        self._validate_input(X, y)

        n = len(X)
        #calculate the average distance of the points in the same cluster
        A = np.array([np.mean(cdist([X[i]], X[y == y[i]])) for i in range(n)]) 

        #calculate the average distance of the points in the nearest cluster
        B = np.array([np.min([np.mean(cdist([X[i]], X[y == label])) for label in set(y) - {y[i]}]) for i in range(n)]) 
                              

        # Handle case where a cluster has only one point
        B[np.isnan(B)] = 0

        # Calculate silhouette scores
        silhouette_scores = (B - A) / np.maximum(A, B)

        # Handle cases where silhouette score is not defined
        silhouette_scores[np.isnan(silhouette_scores)] = 0

    
        if vectorized:
            #Calculate distances efficiently
            distances = squareform(pdist(X, 'euclidean'))

            # Initialize storing arrays
            intra_cluster_dist = np.zeros(X.shape[0])
            nearest_cluster_dist = np.inf * np.ones(X.shape[0])

            # Calculate intra-cluster distances
            for label in np.unique(y):
                mask = (y == label)
                intra_cluster_dist[mask] = distances[mask][:, mask].sum(axis=1) / (mask.sum() - 1)

            # Calculate nearest-cluster distances
            for label in np.unique(y):
                mask = (y == label)
                other_distances = distances[mask][:, ~mask]
                if other_distances.size > 0:
                    nearest_cluster_dist[mask] = np.minimum(other_distances.mean(axis=1), nearest_cluster_dist[mask])

            # Handle edge case where a cluster has only one point
            intra_cluster_dist[np.isnan(intra_cluster_dist)] = 0
            nearest_cluster_dist[np.isnan(nearest_cluster_dist)] = 0

            # Calculate silhouette scores
            silhouette_scores = (nearest_cluster_dist - intra_cluster_dist) / np.maximum(intra_cluster_dist, nearest_cluster_dist)

            # Handle cases where silhouette score is not defined
            silhouette_scores[np.isnan(silhouette_scores)] = 0

        return silhouette_scores
