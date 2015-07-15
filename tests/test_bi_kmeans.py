import numpy as np
import pyclust

import treelib

d1 = np.random.uniform(low=-2, high=2, size=(20,2))
d2 = np.random.uniform(low=2, high=4, size=(10,2))
d3 = np.random.uniform(low=-4, high=-2, size=(10,2))
da = np.vstack((d1,d2,d3))


def test_bisect3(d):
    bkm = pyclust.BisectKMeans(n_clusters=4)

    bkm.fit(d)

    print(bkm.labels_)

    leaf_nodes = bkm.tree_.leaves()
    for node in leaf_nodes:
        print(node.data)


    bkm.tree_.show(line_type='ascii')

    for node in bkm.tree_.all_nodes():
        print(node.data)


    print("Orig: ", np.unique(bkm.labels_))
    print("Cut2: ", np.unique(bkm.cut(2)))
    print("Cut3: ", np.unique(bkm.cut(3)))
    print("Cut4: ", np.unique(bkm.cut(4)))


test_bisect3(da)
