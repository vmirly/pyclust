import numpy as np
import scipy, scipy.spatial

from . import _kmeans as kmeans



def _compute_gram_matrix(X, kern_type, sigma_sq=2.0):
    """
    """
    if kern_type == 'rbf':
        pairwise_dist = scipy.spatial.distance.pdist(X, metric='seuclidean')
        pairwise_dist = scipy.spatial.distance.squareform(pairwise_dist)
        gram_matrix = np.exp( - pairwise_dist / sigma_sq)
    else:
        pass

    return(gram_matrix)



def _kernelized_dist2centers(K, wmemb, i):
    """ Computin the distance in transformed feature space to 
         cluster centers.
 
        K is the kernel gram matrix.
        wmemb contains cluster assignment. {0,1}

        Assume j is the cluster id:
        ||phi(x_i) - Phi_center_j|| = K_ii - 2 sum w_jh K_ih + 
                                      sum_r sum_s w_jr w_js K_rs
    """
    n_samples = K.shape[0]
    n_clusters = wmemb.shape[1]
    
    kdist = np.ones(shape=n_clusters, dtype=float)
    for j in range(n_clusters):
        kdist[j] = kdist[j] - 2 * np.sum(wmemb[:,j] * K[i,:])
        for r in range(n_samples):
            if wmemb[r,j] == 1:
                kdist[j] += np.sum(wmemb[:,j] * K[r,:])

    return(kdist)



def _kernelized_distance_rbf(x_i, x_j, sigma_sq=2.0):
    """
    """
    dx = x_i - x_j
    dx_sq = np.sum(dx * dx)
    return(np.exp(-dx_sq / sigma_sq))


class KernelKMeans(object):
    """
    """

    def __init__(self, n_clusters=2, kernel='linear', n_trials=10, max_iter=100):
        assert (kernel in ['linear', 'rbf'])
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
