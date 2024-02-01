import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 200):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        ## Initialize the attributes

        self._validate_initial_parameters(k, tol, max_iter)
        self.num_clusters = k
        self.tolerance = tol
        self.max_iterations = max_iter
        self.centroids = None
        self.labels = None
        self.error = None

    def _validate_initial_parameters(self, k, tol, max_iter):
        if not isinstance(k, int) or k <= 0:
            raise ValueError("k must be a positive integer")
        if not isinstance(tol, float) or tol <= 0:
            raise ValueError("tol must be a positive float")
        if not isinstance(max_iter, int) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer")

    def _validate_data(self, data):
        if not isinstance(data, np.ndarray) or data.ndim != 2:
            raise ValueError("data must be a 2D numpy array")
        if data.shape[0] < self.num_clusters:
            raise ValueError("data must have more rows than the number of clusters")
        if data.shape[1] == 0:
            raise ValueError("data must have at least one column")
        if self.centroids is not None and data.shape[1] != self.centroids.shape[1]:
            raise ValueError("data must have the same number of columns/dimensions as the centroids")
    
    


    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        ## handle errors

        self._validate_data(mat)

       
        ## Initialize the centroids
        self.centroids = mat[np.random.choice(mat.shape[0], self.num_clusters, replace=False)]
        ## Initialize the error
        self.error = np.inf
        ## Initialize the iteration
        self.iteration = 0

        while self.error > self.tolerance and self.iteration < self.max_iterations:
            self.labels = np.argmin(cdist(mat, self.centroids), axis=1)    #assign points to the nearest centroid
            new_centroids = []
            for i in range(self.num_clusters):
                cluster_points = mat[self.labels == i]
                if len(cluster_points) > 0:
                    new_centroids.append(cluster_points.mean(axis=0))    #calculate the new centroids

                else:
                    #handle empty clusters
                    new_centroids.append(mat[np.random.choice(mat.shape[0])])
            new_centroids = np.array(new_centroids)
            self.error = sum(np.sum((mat[self.labels == i] - self.centroids[i]) ** 2) for i in range(self.num_clusters)) #calculate the error
            self.centroids = new_centroids     #update the centroids

            self.iteration += 1




    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        self._validate_data(mat)
        #ensure that the model has been fit
        if self.centroids is None:
            raise ValueError("Model has not been fit yet")
        
        return np.argmin(cdist(mat, self.centroids), axis=1)


    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model. You can either do this by storing the
        original dataset or recording it following the end of model fitting.

        outputs:
            float
                the squared-mean error of the fit model
        """
        #ensure that the model has been fit
        if self.centroids is None:
            raise ValueError("Model has not been fit yet")
        
        return self.error if self.error is not None else np.inf


    def get_centroids(self) -> np.ndarray:
        """
        Returns the centroid locations of the fit model.

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        #ensure that the model has been fit
        if self.centroids is None:
            raise ValueError("Model has not been fit yet")
        
        return self.centroids if self.centroids is not None else np.array([])

