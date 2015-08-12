import scipy, scipy.spatial
import numpy as np

def _cal_silhouette_score(X, y, sample_size, metric):
    """
    """
    n_samples = X.shape[0]
    if sample_size is not None:
       assert(1 < sample_size < n_samples)
       rand_inx = np.random.choice(X.shape[0], size=sample_size)
       X_samp, y_samp = X[rand_inx], y[rand_inx]

    else:
       X_samp, y_samp = X, y

    n_clust_samp = len(np.unique(y_samp))
    pair_dist = scipy.spatial.distance.pdist(X_samp, metric=metric)

    return(pair_dist)

class Silhouette(object):
    """
    """
    def __init__(self):
        self.n_labels_ = None

    def score(self, X, labels, sample_size=None, metric='euclidean'):
        self.score = _cal_silhouette_score(X, labels, sample_size, metric)

        return(self.score)
