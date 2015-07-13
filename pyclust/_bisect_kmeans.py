import warnings

import numpy as np

from . import _kmeans



def _select_cluster_2_split(membs, sse_arr):
    if np.any(sse_arr > 0):
        clust_id = np.argmax(sse_arr)
        memb_ids = np.where(membs == clust_id)[0]
        return(memb_ids)
    else:
        return(np.range(membs.shape[0]))


def _bisect_kmeans(X, n_clusters, n_trials, max_iter, sse_arr):
    """
    """
    membs = np.empty(shape=X.shape[0], dtype=int)
    km = _kmeans.KMeans(n_clusters=2, n_trials=n_trials, max_iter=max_iter)
    for i in range(n_clusters-1):
        sel_ids = _select_cluster_2_split(, sse_arr)
        X_sub = X[sel_ids,:]
        centers,  km.fit(X_sub)


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
