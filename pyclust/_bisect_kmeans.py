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



def _cut_tree(tree, n_clusters, membs):
    """ Cut the tree to get desired number of clusters as n_clusters
            2 <= n_desired <= n_clusters
    """
    assert(n_clusters >= 2)
    assert(n_clusters <= len(tree.leaves()))

    node_set = set(tree.children(0))
    cut_set = set()
    for i in range(2,n_clusters+1):
        iter_nodes = node_set.copy()
        for n in iter_nodes:
            #print(i, n.data['ilev'], n.data['label'])
            node_set.remove(n)
            if n.data['ilev'] is None:
                cut_set.add(n)
            elif n.data['ilev'] == i:
                nid = n.identifier
                node_set = cut_set.union(set(tree.children(nid)))
                #print(nid, tree.children(nid), node_set)

            if i==(n_clusters):
                cut_set.add(n)
   
    conv_membs = membs.copy()
    for node in cut_set:
        nid = node.identifier
        sub_leaves = tree.leaves(nid)
        for leaf in sub_leaves:
            indx = np.where(conv_membs == leaf)[0]
            conv_membs[indx] = nid

    return(conv_membs)


def _add_tree_node(tree, label, ilev, X=None, center=None, sse=None, parent=None):
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
        'sse'   : sse,
        'ilev'  : None 
    }
    if (parent is None):
        tree.create_node(label, label, data=datadict)
    else:
        tree.create_node(label, label, parent=parent, data=datadict)
        tree.get_node(parent).data['ilev'] = ilev

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
    tree = _add_tree_node(tree, 0, ilev=0, X=X) 

    km = _kmeans.KMeans(n_clusters=2, n_trials=n_trials, max_iter=max_iter, tol=tol)
    for i in range(1,n_clusters):
        sel_clust_id,sel_memb_ids = _select_cluster_2_split(membs, tree)
        X_sub = X[sel_memb_ids,:]
        km.fit(X_sub)

        print("Bisecting Step %d    :"%i, sel_clust_id, km.sse_arr_, km.centers_)
        ## Updating the clusters & properties
        #sse_arr[[sel_clust_id,i]] = km.sse_arr_
        #centers[[sel_clust_id,i]] = km.centers_
        tree = _add_tree_node(tree, 2*i-1, i, center=km.centers_[0], \
                             sse=km.sse_arr_[0], parent= sel_clust_id)
        tree = _add_tree_node(tree, 2*i,   i, center=km.centers_[1], \
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


    def cut(self, n_desired):
        """
        """
        return(_cut_tree(self.tree_, n_desired, self.labels_))
