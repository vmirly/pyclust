import warnings

import numpy as np


def _kmeans_init(X, n_clusters):
    """ Initialize k=n_cluster centroids randomly
    """

def _assign_clusters(X, centeroids):
    """ Assignment Step:
	   assign each point to the closet cluster center
    """

def _update_centers(X, memb):
    """ Update Cluster Centers:
	   calculate the mean of feature vectors for each cluster
    """

def _kmeans_run(X, centroids, max_iter):
    """ Run a single trial of k-means clustering
	on dataset X, and starting centroids
    """

class KMeans(object):
    """
	KMeans Clustering

	Parameters
	-------
	   n_cluster

	   n_trial : int, default 10

	   max_iter : int, default 100

	Attibutes
	-------

	   

	Methods
	------- 
	   fit()
	   predict()
	   fit_predict()
    """

    def __init__(self, n_clusters=2, n_trial=10, max_iter=100):
	
	self.n_clusters = n_clusters
	self.n_trial = ntrial
	self.max_iter = max_iter


    def _k_init(self):
	self.size_, self.dim_ = X.shape[0], X.shape[1]
	cent_idx_ = np.random.choice(self.size_, replace=False, size=self.n_clusters)

	self.centroids_ = X[cent_idx_, :]

    def fit(self, X, y=None):
	""" Apply KMeans Clustering
	      X: dataset with feature vectors
	"""

	self.size_, self.n_dim_ = X.shape[0], X.shape[1]

	centers_ = np.empty(shape=(self.n_trial,seld.n_dim_), dtype=float)
	self.sum_sqr_err_ = np.empty(shape=self.n_trial, dtype=float)
	for i in range(self.n_trial):
	    init_centers = self._kmeans_init()
	    self.centroids_, self.labels_ = _kmeans()
