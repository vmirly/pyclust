import numpy as np
import scipy, scipy.spatial

from . import _kmeans as kmeans



def _compute_gram_matrix(X, kern_type, sigma_sq=2.0):
    """
    """
    if kern_type == 'rbf':
        pairwise_dist = scipy.spatial.distance.pdist(X, metric='sqeuclidean')
        pairwise_dist = scipy.spatial.distance.squareform(pairwise_dist)
        gram_matrix = np.exp( - pairwise_dist / sigma_sq)
    else:
        pass

    return(gram_matrix)



def _kernelized_dist2centers(K, n_clusters, wmemb, kernel_dist):
    """ Computin the distance in transformed feature space to 
         cluster centers.
 
        K is the kernel gram matrix.
        wmemb contains cluster assignment. {0,1}

        Assume j is the cluster id:
        ||phi(x_i) - Phi_center_j|| = K_ii - 2 sum w_jh K_ih + 
                                      sum_r sum_s w_jr w_js K_rs
    """
    n_samples = K.shape[0]
    
    for j in range(n_clusters):
        memb_j = np.where(wmemb == j)[0]
        size_j = memb_j.shape[0]

        KK = K[memb_j][:, memb_j] 
        kernel_dist[:,j] = np.sum(KK)/(size_j*size_j)
        kernel_dist[:,j] -= 2 * np.sum(K[:, memb_j], axis=1) / size_j



def _fit_kernelkmeans(K, n_clusters, max_iter):
    """
    """
    n_samples = K.shape[0]
    labels_old = np.random.randint(n_clusters, size=n_samples)
    kdist = np.empty(shape=(n_samples, n_clusters), dtype=float)
    for it in range(max_iter):
        kdist.fill(0)
        _kernelized_dist2centers(K, n_clusters, labels_old, kdist)
        labels = np.argmin(kdist, axis=1)


    return(labels)



class KernelKMeans(object):
    """
    """

    def __init__(self, n_clusters=2, kernel_type='linear', n_trials=10, max_iter=100):
        assert (kernel in ['linear', 'rbf'])
        self.n_clusters  = n_clusters
        self.kernel_type = kernel_type
        self.n_trials    = n_trials
        self.max_iter    = max_iter

    def fit(self, X):
        """
        """
        self.kernel_matrix_ = _compute_gram_matrix(X, self.kernel_type)
        self.labels_ = _fit_kernelkmeans(self.kernel_matrix_, self.n_clusters, self.max_iter)
        self.converged = True

    def fit_predict(self, X):
        """
        """
        if not self.converged:
            raise("Model is not fit yet!")

        self.fit(X)
        return(self.labels_)
