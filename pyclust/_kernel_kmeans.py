import numpy as np
import scipy, scipy.spatial

from . import _kmeans as kmeans



def _kernelized_distance(x_i, wcent):
    """
    """


def _assign_clusters(X, centers):
    """ Assignment Step:
           assign each point to the closet cluster center

        center_j = sum_h w_hj phi(x_h)
    """
    dist2cents = scipy.spatial.distance.cdist(X, centers, metric='euclidean')
    membs = np.argmin(dist2cents, axis=1)

    return(membs)



class KernelKmeans(object):
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
