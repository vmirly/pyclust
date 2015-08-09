### Theoretical & Algorithmic Discussion 
###  with Emad Zahedi
import numpy as np
import scipy, scipy.spatial



def _compute_gram_matrix(X, kernel_type, params):
    """
    """
    if kernel_type == 'rbf':
        if 'gamma' in params:
            gamma = params['gamma']
        else:
            gamma = 1.0 / X.shape[1]

        pairwise_dist = scipy.spatial.distance.pdist(X, metric='sqeuclidean')
        pairwise_dist = scipy.spatial.distance.squareform(pairwise_dist)
        gram_matrix = np.exp( - gamma * pairwise_dist )

        np.fill_diagonal(gram_matrix, 1)
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

        K_sub_j = K[memb_j][:, memb_j]
         
        kernel_dist[:,j] = 1 + np.sum(K_sub_j) /(size_j*size_j)
        kernel_dist[:,j] -= 2 * np.sum(K[:, memb_j], axis=1) / size_j

    return


def _fit_kernelkmeans(K, n_clusters, n_trials, max_iter, converge_tol=0.001):
    """
    """
    n_samples = K.shape[0]
    kdist = np.empty(shape=(n_samples, n_clusters), dtype=float)
    within_distances = np.empty(shape=n_clusters, dtype=float)

    best_within_distances = np.infty
    for i in range(n_trials):
        membs_prev = np.random.randint(n_clusters, size=n_samples)

        for it in range(max_iter):
            kdist.fill(0)
            _kernelized_dist2centers(K, n_clusters, membs_prev, kdist)

            membs_curr = np.argmin(kdist, axis=1)
            membs_changed_ratio = float(np.sum((membs_curr - membs_prev) != 0)) / n_samples

            if membs_changed_ratio < converge_tol:
                break

            membs_prev = membs_curr

        for j in range(n_clusters):
            within_distances[j] = np.sum(kdist[np.where(membs_curr == j)[0], j])
        if best_within_distances > within_distances.sum():
            best_within_distances = within_distances.sum()
            best_labels = membs_curr

    return(it, best_labels)



class KernelKMeans(object):
    """
    """

    def __init__(self, n_clusters=2, kernel='linear', params={}, n_trials=10, max_iter=100):
        assert (kernel in ['linear', 'rbf'])
        self.n_clusters  = n_clusters
        self.kernel_type = kernel
        self.n_trials    = n_trials
        self.max_iter    = max_iter

        self.kernel_params = params
        self.kernel_matrix_ = None


    def _set_kernel_matrix(self, X=None, kernel_matrix=None):
        """
        """
        if self.kernel_matrix_ is None:
            if kernel_matrix is None:
                if X is None:
                    raise("Either X or kernel_matrix is needed!")
                self.kernel_matrix_ = _compute_gram_matrix(X, self.kernel_type, self.kernel_params)
            else:
                self.kernel_matrix_ = kernel_matrix


    def fit(self, X, kernel_matrix=None):
        """
        """
        self._set_kernel_matrix(X, kernel_matrix)
        self.n_iter_, self.labels_ = _fit_kernelkmeans(self.kernel_matrix_, self.n_clusters, self.n_trials, self.max_iter)


    def fit_predict(self, X):
        """
        """
        self.fit(X)
        return(self.labels_)


############### Global Kernel K-Means ####################


def _fit_global_kernelkmeans(K, n_clusters, max_iter, converge_tol=0.001):
    """
    """
    n_samples = K.shape[0]
    kdist = np.empty(shape=(n_samples, n_clusters), dtype=float)
    within_distances = np.empty(shape=n_clusters, dtype=float)

    best_within_distances = np.infty
    for i in range(n_samples):
        membs_prev = np.random.randint(n_clusters, size=n_samples)

        for it in range(max_iter):
            kdist.fill(0)
            _kernelized_dist2centers(K, n_clusters, membs_prev, kdist)

            membs_curr = np.argmin(kdist, axis=1)
            membs_changed_ratio = float(np.sum((membs_curr - membs_prev) != 0)) / n_samples

            if membs_changed_ratio < converge_tol:
                break

            membs_prev = membs_curr

        for j in range(n_clusters):
            within_distances[j] = np.sum(kdist[np.where(membs_curr == j)[0], j])
        if best_within_distances > within_distances.sum():
            best_within_distances = within_distances.sum()
            best_labels = membs_curr

    return(it, best_labels)

class GlobalKernelKMeans(object):
    """
    """
    def __init__(self, n_clusters=3, kernel='linear', params={}, n_trials=10, max_iter=100):
        self.n_clusters  = n_clusters
        self.kernel_type = kernel
        self.n_trials    = n_trials
        self.max_iter    = max_iter

        self.kernel_params = params
        self.kernel_matrix_ = None

    def fit(self, X):
        """
        """
        if self.kernel_matrix_ is None:
            self.kernel_matrix_ = _compute_gram_matrix(X, self.kernel_type, self.kernel_params)
        

    def refit(self, n_clusters):
        """ Extend clustering to a larger number of clusters
        """
        if n_clusters <= self.n_clusters:
            pass
        else:
            pass
            
