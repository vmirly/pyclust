import numpy as np
import scipy, scipy.spatial

from . import _kmeans as kmeans



def _compute_gram_matrix(X, kern_type='rbf', sigma_sq=2.0):
    """
    """
    if kern_type == 'rbf':
        pairwise_dist = scipy.spatial.distance.pdist(X, metric='seuclidean')
        gram_matrix = np.exp( - pairwise_dist / sigma_sq)
    else:
        print("OOO")
        print(kern_type)

    return(gram_matrix)

def _kernelized_distance_rbf(x_i, x_j, sigma_sq_2=2.0):
    """
    """
    dx = x_i - x_j
    dx_sq = np.sum(dx * dx)
    return(np.exp(-dx_sq / sigma_sq_2))


class KernelKMeans(object):
    """
    """

    def __init__(self, n_clusters=2, kernel='rbf', n_trials=10, max_iter=100):
        self.n_clusters = n_clusters
        self.kernel     = kernel
        self.n_trials   = n_trials
        self.max_iter   = max_iter

    def fit(self, X):
        """
        """
        pass

    def fit_predict(self, X):
        """
        """
        pass
