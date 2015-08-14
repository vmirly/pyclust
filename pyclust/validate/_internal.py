import scipy, scipy.spatial
import numpy as np

import sys

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

    pair_dist_matrix = scipy.spatial.distance.squareform(pair_dist)

    ## Compute average intra-cluster distances 
    ## for each of the selected samples
    a_arr = _intra_cluster_distances(pair_dist_matrix, y_samp, np.arange(y_samp.shape[0]))

    ## Compute Avg. Distabces to the Neighboring Clusters
    b_arr =  _neighboring_cluster_distances(pair_dist_matrix, y_samp, np.arange(y_samp.shape[0]))

    comb_arr = np.vstack((a_arr, b_arr))
    return((b_arr-a_arr)/np.max(comb_arr, axis=0), a_arr, b_arr)


def _intra_cluster_distances(dist_matrix, y, ilist):
    """
    """
    n_samples = y.shape[0]
    if type(ilist) == type(list()):
        n_inx = len(ilist)
    elif type(ilist) == type(np.array([])):
        n_inx = ilist.shape[0]
    else:
        raise Exception("ilist must be iterable!")

    mean_intra_distances = np.empty(shape=n_inx, dtype=float)
    for i,inx in enumerate(ilist):
        mask = y == y[inx]
        mask[inx] = False
        if np.sum(mask)>0:
            mean_intra_distances[i] = np.mean(dist_matrix[inx][mask]) #/(np.sum(mask) - 1.0)
        else:
            mean_intra_distances[i] = 0.0
        #sys.stderr.write("mask size: %d  AvgIntraDist %f\n"%(np.sum(mask), mean_intra_distances[i]))
    return(mean_intra_distances) 



def _neighboring_cluster_distances(dist_matrix, y, ilist):
    """
    """
    n_samples = y.shape[0]
    if type(ilist) == type(list()):
        n_inx = len(ilist)
    elif type(ilist) == type(np.array([])):
        n_inx = ilist.shape[0]
    else:
        raise Exception("ilist must be iterable!")

    min_clust_distances = np.empty(shape=n_inx, dtype=float)
    for i,inx in enumerate(ilist):
        y_inx = y[inx]
        dist_row_inx = dist_matrix[inx]
        dist_2_clusters = [np.mean(dist_row_inx[y == j]) for j in set(y) if not j == y_inx]
        min_clust_distances[i] = np.min(dist_2_clusters)

        #sys.stderr.write("mask size: %d  Nearest Cluster %f\n"%(np.sum(mask),min_clust_distances[i]))
    return(min_clust_distances)


class Silhouette(object):
    """
    """
    def __init__(self):
        self.n_labels_ = None

    def score(self, X, labels, sample_size=None, metric='euclidean'):
        self.sample_scores, self.a_, self.b_ = _cal_silhouette_score(X, labels, sample_size, metric)
        self.score = np.mean(self.sample_scores)

        return(self.score, self.a_, self.b_)
