import numpy as np
import scipy.spatial

def _kmeans_init(X, n_clusters, method='balanced', rng=None):
    """ Initialize k=n_clusters centroids randomly
    """
    n_samples = X.shape[0]
    if rng is None:
        cent_idx = np.random.choice(n_samples, replace=False, size=n_clusters)
    else:
        #print('Generate random centers using RNG')
        cent_idx = rng.choice(n_samples, replace=False, size=n_clusters)
    
    centers = X[cent_idx,:]
    mean_X = np.mean(X, axis=0)
    
    if method == 'balanced':
        centers[n_clusters-1] = n_clusters*mean_X - np.sum(centers[:(n_clusters-1)], axis=0)
    
    return (centers)


def _assign_clusters(X, centers):
    """ Assignment Step:
           assign each point to the closet cluster center
    """
    dist2cents = scipy.spatial.distance.cdist(X, centers, metric='seuclidean')
    membs = np.argmin(dist2cents, axis=1)

    return(membs)

def _cal_dist2center(X, center):
    """ Calculate the SSE to the cluster center
    """
    dmemb2cen = scipy.spatial.distance.cdist(X, center.reshape(1,X.shape[1]), metric='seuclidean')
    return(np.sum(dmemb2cen))

def _update_centers(X, membs, n_clusters):
    """ Update Cluster Centers:
           calculate the mean of feature vectors for each cluster
    """
    centers = np.empty(shape=(n_clusters, X.shape[1]), dtype=float)
    sse = np.empty(shape=n_clusters, dtype=float)
    for clust_id in range(n_clusters):
        memb_ids = np.where(membs == clust_id)[0]

        if memb_ids.shape[0] == 0:
            memb_ids = np.random.choice(X.shape[0], size=1)
            #print("Empty cluster replaced with ", memb_ids)
        centers[clust_id,:] = np.mean(X[memb_ids,:], axis=0)
        
        sse[clust_id] = _cal_dist2center(X[memb_ids,:], centers[clust_id,:]) 
    return(centers, sse)



def _kmeans_run(X, n_clusters, max_iter, tol):
    """ Run a single trial of k-means clustering
        on dataset X, and given number of clusters
    """
    membs = np.empty(shape=X.shape[0], dtype=int)
    centers = _kmeans_init(X, n_clusters)

    sse_last = 9999.9
    n_iter = 0
    for it in range(1,max_iter):
        membs = _assign_clusters(X, centers)
        centers,sse_arr = _update_centers(X, membs, n_clusters)
        sse_total = np.sum(sse_arr)
        if np.abs(sse_total - sse_last) < tol:
            n_iter = it
            break
        sse_last = sse_total

    return(centers, membs, sse_total, sse_arr, n_iter)


def _kmeans(X, n_clusters, max_iter, n_trials, tol):
    """ Run multiple trials of k-means clustering,
        and outputt he best centers, and cluster labels
    """
    n_samples, n_features = X.shape[0], X.shape[1]

    centers_best = np.empty(shape=(n_clusters,n_features), dtype=float)
    labels_best  = np.empty(shape=n_samples, dtype=int)
    for i in range(n_trials):
        centers, labels, sse_tot, sse_arr, n_iter  = _kmeans_run(X, n_clusters, max_iter, tol)
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


class KMeans(object):
    """
        KMeans Clustering

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

    def __init__(self, n_clusters=2, n_trials=10, max_iter=100, tol=0.001):
        
        self.n_clusters = n_clusters
        self.n_trials = n_trials
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X):
        """ Apply KMeans Clustering
              X: dataset with feature vectors
        """
        self.centers_, self.labels_, self.sse_arr_, self.n_iter_ = \
              _kmeans(X, self.n_clusters, self.max_iter, self.n_trials, self.tol)


    def fit_predict(self, X):
        """ Apply KMeans Clustering, 
            and return cluster labels
        """
        self.fit(X)
        return(self.labels_)
