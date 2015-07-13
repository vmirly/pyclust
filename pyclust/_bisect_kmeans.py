import warnings

import numpy as np

from . import _kmeans



def _select_cluster_2_split(membs, sse_arr):
    if np.any(sse_arr > 0):
        clust_id = np.argmax(sse_arr)
        memb_ids = np.where(membs == clust_id)[0]
        return(clust_id,memb_ids)
    else:
        return(0,np.arange(membs.shape[0]))


def _bisect_kmeans(X, n_clusters, n_trials, max_iter):
    """
    """
    membs = np.empty(shape=X.shape[0], dtype=int)
    centers = np.empty(shape=(n_clusters,X.shape[1]), dtype=float)
    sse_arr = -1.0*np.ones(shape=n_clusters, dtype=float)

    km = _kmeans.KMeans(n_clusters=2, n_trials=n_trials, max_iter=max_iter)
    for i in range(1,n_clusters):
        sel_clust_id,sel_memb_ids = _select_cluster_2_split(membs, sse_arr)
        X_sub = X[sel_memb_ids,:]
        km.fit(X_sub)

        ## Updating the clusters & properties
        sse_arr[[sel_clust_id,i]] = km.sse_arr_
        membs[sel_memb_ids] = km.labels_

    return(centers, membs, sse_arr)

class BisectKMeans(object):
    """ 
        KMeans Clustering

        Parameters
        -------

        Attibutes
        -------

           

        Methods
        ------- 
           fit()
           predict()
           fit_predict()
    """
    def __init__(self, n_clusters=2, n_trials=10, max_iter=100):
        assert n_clusters >= 2, 'n_clusters should be >= 2'
        self.n_clusters = n_clusters
        self.n_trials = n_trials
        self.max_iter = max_iter

        ## place to save the SSE values
        self.sse_arr_ = -1*np.ones(shape=n_clusters, dtype=float)

    def fit(self, X, y=None):
        """
        """
        _bisect_kmeans(X, self.n_clusters, self.n_trials, self.max_iter)
