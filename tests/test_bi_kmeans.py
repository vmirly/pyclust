import numpy as np
import pyclust

import treelib

d1 = np.random.uniform(low=-2, high=2, size=(20,2))
d2 = np.random.uniform(low=2, high=4, size=(10,2))
d3 = np.random.uniform(low=-4, high=-2, size=(10,2))
da = np.vstack((d1,d2,d3))


def test_bisect3():
    bkm = pyclust.BisectKMeans(n_clusters=4)

    bkm.fit(da)

    print(bkm.labels_)

    leaf_nodes = bkm.tree_.leaves()
    for node in leaf_nodes:
        print(node.data)


    bkm.tree_.show(line_type='ascii')

    print("Orig: ", np.unique(bkm.labels_))
    print("Cut2: ", np.unique(bkm.cut(2)[0]))
    print("Cut3: ", np.unique(bkm.cut(3)[0]))
    print("Cut4: ", np.unique(bkm.cut(4)[0]))


def test_bisect10():
    nsamp = 20
    da = np.random.multivariate_normal(mean=(-2,0), cov=[[0.12,0],[0,0.12]], size=nsamp)
    ya = np.ones(shape=nsamp, dtype=int)

    db = np.random.multivariate_normal(mean=(-3,1), cov=[[0.08,0.05],[0.05,0.08]], size=nsamp)
    yb = 2*np.ones(shape=nsamp, dtype=int)

    dc = np.random.multivariate_normal(mean=(-3,-1), cov=[[0.08,-0.05],[-0.05,0.08]], size=nsamp)
    yc = 3*np.ones(shape=nsamp, dtype=int)

    dd = np.random.multivariate_normal(mean=(-1,1), cov=[[0.08,-0.05],[-0.05,0.08]], size=nsamp)
    yd = 4*np.ones(shape=nsamp, dtype=int)

    de = np.random.multivariate_normal(mean=(-1,-1), cov=[[0.08,0.05],[0.05,0.08]], size=nsamp)
    ye = 5*np.ones(shape=nsamp, dtype=int)

    df = np.random.multivariate_normal(mean=(3,0.6), cov=[[0.05,0.0],[0.0,0.05]], size=nsamp)
    yf = 6*np.ones(shape=nsamp, dtype=int)

    dg = np.random.multivariate_normal(mean=(3,-0.6), cov=[[0.05,0.0],[0.0,0.05]], size=nsamp)
    yg = 7*np.ones(shape=nsamp, dtype=int)

    X = np.vstack((da, db, dc, dd, de, df, dg))
    y = np.hstack((ya, yb, yc, yd, ye, yf, yg))


    bkm = pyclust.BisectKMeans(n_clusters=10)

    bkm.fit(X)

    leaf_nodes = bkm.tree_.leaves()
    for node in leaf_nodes:
        print(node.data)

    bkm.tree_.show(line_type='ascii')

    print("Orig: ", np.unique(bkm.labels_))
    print("Cut2: ", np.unique(bkm.cut(2)[0]))
    print("Cut3: ", np.unique(bkm.cut(3)[0]))
    print("Cut4: ", np.unique(bkm.cut(4)[0]))

    print("Cut5: ", np.unique(bkm.cut(5)[0]))
    print("Cut6: ", np.unique(bkm.cut(6)[0]))
    print("Cut7: ", np.unique(bkm.cut(7)[0]))

test_bisect10()

test_bisect3()
