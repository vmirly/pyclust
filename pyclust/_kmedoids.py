import numpy as np
import scipy.spatial

from . import _kmeans as kmeans



def _update_centers(X, membs, n_clusters, distance):
    """ Update Cluster Centers:
           calculate the mean of feature vectors for each cluster.

        distance can be a string or callable.
    """
    centers = np.empty(shape=(n_clusters, X.shape[1]), dtype=float)
    sse = np.empty(shape=n_clusters, dtype=float)
    for clust_id in range(n_clusters):
        memb_ids = np.where(membs == clust_id)[0]
        X_clust = X[memb_ids,:]

        dist = np.empty(shape=memb_ids.shape[0], dtype=float)
        for i,x in enumerate(X_clust):
           dist[i] = np.sum(scipy.spatial.distance.cdist(X_clust, np.array([x]), distance))

        inx_min = np.argmin(dist)
        centers[clust_id,:] = X_clust[inx_min,:]
        sse[clust_id] = dist[inx_min]
    return(centers, sse)



def _kmedoids_run(X, n_clusters, distance, max_iter, tol, rng):
    """ Run a single trial of k-medoids clustering
        on dataset X, and given number of clusters
    """
    membs = np.empty(shape=X.shape[0], dtype=int)
    centers = kmeans._kmeans_init(X, n_clusters, method='', rng=rng)

    sse_last = 9999.9
    n_iter = 0
    for it in range(1,max_iter):
        membs = kmeans._assign_clusters(X, centers)
        centers,sse_arr = _update_centers(X, membs, n_clusters, distance)
        sse_total = np.sum(sse_arr)
        if np.abs(sse_total - sse_last) < tol:
            n_iter = it
            break
        sse_last = sse_total

    return(centers, membs, sse_total, sse_arr, n_iter)


def _kmedoids(X, n_clusters, distance, max_iter, n_trials, tol, rng):
    """ Run multiple trials of k-medoids clustering,
        and output he best centers, and cluster labels
    """
    n_samples, n_features = X.shape[0], X.shape[1]

    centers_best = np.empty(shape=(n_clusters,n_features), dtype=float)
    labels_best  = np.empty(shape=n_samples, dtype=int)
    for i in range(n_trials):
        centers, labels, sse_tot, sse_arr, n_iter  = \
               _kmedoids_run(X, n_clusters, distance, max_iter, tol, rng)
        if i==0:
            sse_tot_best = sse_tot
            sse_arr_best = sse_arr
            n_iter_best = n_iter
            centers_best = centers.copy()
            labels_best  = labels.copy()
        if sse_tot < sse_tot_best:
            sse_tot_best = sse_tot
            sse_arr_best = sse_arr
            n_iter_best = n_iter
            centers_best = centers.copy()
            labels_best  = labels.copy()

    return(centers_best, labels_best, sse_arr_best, n_iter_best)


class KMedoids(object):
    """
        KMedoids Clustering

        K-medoids clustering take the cluster centroid as the medoid of the data points,
         as opposed to the average of data points in a cluster. As a result, K-medoids 
         gaurantees that the cluster centroid is among the cluster members.

        The medoid is defined as the point that minimizes the total within-cluster distances.

        K-medoids is more robust to outliers (the reason for this is similar to why 
         median is more robust to mean).

        K-medoids is computationally more expensive, since it involves computation of all
         the pairwise distances in a cluster.


        Parameters
        -------
           n_clusters: number of clusters (default = 2)
           n_trials: number of trial random centroid initialization (default = 10)
           max_iter: maximum number of iterations (default = 100)
           tol: tolerance (default = 0.0001)


        Attibutes
        -------
           labels_   :  cluster labels for each data item
           centers_  :  cluster centers
           sse_arr_  :  array of SSE values for each cluster
           n_iter_   :  number of iterations for the best trial
           

        Methods
        ------- 
           fit(X): fit the model
           fit_predict(X): fit the model and return the cluster labels
    """

    def __init__(self, n_clusters=2, distance='euclidean', 
                 n_trials=10, max_iter=100, tol=0.001, random_state=None):
        
        self.n_clusters = n_clusters
        self.n_trials = n_trials
        self.max_iter = max_iter
        self.tol = tol
        self.distance = distance
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

    def fit(self, X):
        """ Apply KMeans Clustering
              X: dataset with feature vectors
        """
        self.centers_, self.labels_, self.sse_arr_, self.n_iter_ = \
              _kmedoids(X, self.n_clusters, self.distance, self.max_iter, self.n_trials, self.tol, self.rng)


    def fit_predict(self, X):
        """ Apply KMeans Clustering, 
            and return cluster labels
        """
        self.fit(X)
        return(self.labels_)
