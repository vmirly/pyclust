import numpy as np
import treelib

from . import _kmeans



def _select_cluster_2_split(membs, tree):
    leaf_nodes = tree.leaves()
    num_leaves = len(leaf_nodes)
    if len(leaf_nodes)>1:
        sse_arr = np.empty(shape=num_leaves, dtype=float)
        labels  = np.empty(shape=num_leaves, dtype=int)

        for i,node in enumerate(leaf_nodes):
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
    ## starting from root,
    ## a node is added to the cut_set or 
    ## its children are added to node_set
    assert(n_clusters >= 2)
    assert(n_clusters <= len(tree.leaves()))

    cut_centers = dict() #np.empty(shape=(n_clusters, ndim), dtype=float)
    
    for i in range(n_clusters-1):
        if i==0:
            search_set = set(tree.children(0))
            node_set,cut_set = set(), set()
        else:
            search_set = node_set.union(cut_set)
            node_set,cut_set = set(), set()

        if i+2 == n_clusters:
            cut_set = search_set
        else:
            for _ in range(len(search_set)):
                n = search_set.pop()
            
                if n.data['ilev'] is None or n.data['ilev']>i+2:
                    cut_set.add(n)
                else:
                    nid = n.identifier
                    if n.data['ilev']-2==i:
                        node_set = node_set.union(set(tree.children(nid)))
   
    conv_membs = membs.copy()
    for node in cut_set:
        nid = node.identifier
        label = node.data['label']
        cut_centers[label] = node.data['center']
        sub_leaves = tree.leaves(nid)
        for leaf in sub_leaves:
            indx = np.where(conv_membs == leaf)[0]
            conv_membs[indx] = nid

    return(conv_membs, cut_centers)


def _add_tree_node(tree, label, ilev,  X=None, size=None, center=None, sse=None, parent=None):
    """ Add a node to the tree
         if parent is not known, the node is a root

        The nodes of this tree keep properties of each cluster/subcluster:
           size   --> cluster size as the number of points in the cluster
           center --> mean of the cluster
           label  --> cluster label
           sse    --> sum-squared-error for that single cluster
           ilev   --> the level at which this node is split into 2 children
    """
    if size is None:
        size = X.shape[0]
    if (center is None):
        center = np.mean(X, axis=0)
    if (sse is None):
        sse = _kmeans._cal_dist2center(X, center)

    center = list(center)
    datadict = {
        'size'  : size,
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
    centers = dict() #np.empty(shape=(n_clusters,X.shape[1]), dtype=float)
    sse_arr = dict() #-1.0*np.ones(shape=n_clusters, dtype=float)

    ## data structure to store cluster hierarchies
    tree = treelib.Tree()
    tree = _add_tree_node(tree, 0, ilev=0, X=X) 

    km = _kmeans.KMeans(n_clusters=2, n_trials=n_trials, max_iter=max_iter, tol=tol)
    for i in range(1,n_clusters):
        sel_clust_id,sel_memb_ids = _select_cluster_2_split(membs, tree)
        X_sub = X[sel_memb_ids,:]
        km.fit(X_sub)

        #print("Bisecting Step %d    :"%i, sel_clust_id, km.sse_arr_, km.centers_)
        ## Updating the clusters & properties
        #sse_arr[[sel_clust_id,i]] = km.sse_arr_
        #centers[[sel_clust_id,i]] = km.centers_
        tree = _add_tree_node(tree, 2*i-1, i, \
                              size=np.sum(km.labels_ == 0), center=km.centers_[0], \
                              sse=km.sse_arr_[0], parent= sel_clust_id)
        tree = _add_tree_node(tree, 2*i,   i, \
                             size=np.sum(km.labels_ == 1), center=km.centers_[1], \
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


    for n in tree.leaves():
        label = n.data['label']
        centers[label] = n.data['center']
        sse_arr[label] = n.data['sse']

    return(centers, membs, sse_arr, tree)




class BisectKMeans(object):
    """ 
        bisecting KMeans Clustering

        Parameters
        -------
           n_clusters: number of clusters (default = 2)
           n_trials: number of trial random centroid initialization (default = 10)
           max_iter: maximum number of iterations (default = 100)
           tol: tolerance (default = 0.0001)


        Attibutes
        -------
           labels_   :  cluster labels for each data item
           centers_  :  cluster centers
           sse_arr_  :  array of SSE values for each cluster
           n_iter_   :  number of iterations for the best trial
           tree_     :  tree hierarchy of bisecting clusters 

        Methods
        ------- 
           fit(X): fit the model
           fit_predict(X): fit the model and return the cluster labels
    """

    def __init__(self, n_clusters=2, n_trials=10, max_iter=100, tol=0.0001):
        assert n_clusters >= 2, 'n_clusters should be >= 2'
        self.n_clusters = n_clusters
        self.n_trials = n_trials
        self.max_iter = max_iter
        self.tol = tol


    def fit(self, X):
        """
        """
        self.centers_, self.labels_, self.sse_arr_, self.tree_ = \
            _bisect_kmeans(X, self.n_clusters, self.n_trials, self.max_iter, self.tol)


    def fit_predict(self, X):
        """
        """
        self.fit(X)
        return(self.labels_)


    def cut(self, n_desired):
        """
        """
        return(_cut_tree(self.tree_, n_desired, self.labels_))
