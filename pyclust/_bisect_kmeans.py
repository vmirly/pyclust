import warnings

import numpy as np
import treelib

from . import _kmeans



def _select_cluster_2_split(membs, tree):
    leaf_nodes = tree.leaves()
    num_leaves = len(leaf_nodes)
    if len(leaf_nodes)>1:
        sse_arr = np.empty(shape=num_leaves, dtype=float)
        labels  = np.empty(shape=num_leaves, dtype=int)
        i = 0
        for node in leaf_nodes:
            sse_arr[i] = node.data['sse']
            labels[i]  = node.data['label']
        id_max = np.argmax(sse_arr)
        clust_id = labels[id_max]
        memb_ids = np.where(membs == clust_id)[0]
        return(clust_id,memb_ids)
    else:
        return(0,np.arange(membs.shape[0]))


def add_tree_node(tree, label, X=None, center=None, sse=None, parent=None):
    """
    """
    if (center is None):
        center = np.mean(X, axis=0)
    if (sse is None):
        sse = _kmeans._cal_dist2center(X, center)

    center = list(center)
    datadict = {
        'center': center, 
        'label' : label, 
        'sse'   : sse 
    }
    if (parent is not None):
        tree.create_node(label, label, parent=parent, data=datadict)
    else:
        tree.create_node(label, label, data=datadict)

    return(tree)



def _bisect_kmeans(X, n_clusters, n_trials, max_iter, tol):
    """ Apply Bisecting Kmeans clustering
        to reach n_clusters number of clusters
    """
    membs = np.empty(shape=X.shape[0], dtype=int)
    centers = np.empty(shape=(n_clusters,X.shape[1]), dtype=float)
    sse_arr = -1.0*np.ones(shape=n_clusters, dtype=float)

    ## data structure to store cluster hierarchies
    tree = treelib.Tree()
    tree = add_tree_node(tree, 0, X) 

    km = _kmeans.KMeans(n_clusters=2, n_trials=n_trials, max_iter=max_iter, tol=tol)
    for i in range(1,n_clusters):
        sel_clust_id,sel_memb_ids = _select_cluster_2_split(membs, tree)
        X_sub = X[sel_memb_ids,:]
        km.fit(X_sub)

        ## Updating the clusters & properties
        sse_arr[[sel_clust_id,i]] = km.sse_arr_
        centers[[sel_clust_id,i]] = km.centers_
        tree = add_tree_node(tree, 2*i-1, center=km.centers_[0], \
                             sse=km.sse_arr_[0], parent= sel_clust_id)
        tree = add_tree_node(tree, 2*i, center=km.centers_[1], \
                             sse=km.sse_arr_[1], parent= sel_clust_id)

        pred_labels = km.labels_
        pred_labels[np.where(pred_labels == 1)[0]] = 2*i
        pred_labels[np.where(pred_labels == 0)[0]] = 2*i - 1
        #if sel_clust_id == 1:
        #    pred_labels[np.where(pred_labels == 0)[0]] = sel_clust_id
        #    pred_labels[np.where(pred_labels == 1)[0]] = i
        #else:
        #    pred_labels[np.where(pred_labels == 1)[0]] = i
        #    pred_labels[np.where(pred_labels == 0)[0]] = sel_clust_id

        membs[sel_memb_ids] = pred_labels

    return(centers, membs, sse_arr, tree)




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
    def __init__(self, n_clusters=2, n_trials=10, max_iter=100, tol=0.0001):
        assert n_clusters >= 2, 'n_clusters should be >= 2'
        self.n_clusters = n_clusters
        self.n_trials = n_trials
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, X, y=None):
        """
        """
        self.centers_, self.labels_, self.sse_arr_, self.tree_ = \
            _bisect_kmeans(X, self.n_clusters, self.n_trials, self.max_iter, self.tol)
        
