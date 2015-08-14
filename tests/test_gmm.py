import numpy as np
import scipy, scipy.linalg
import sys

import pyclust


s1 = np.array([[0.3, 0.2], [0.7, 0.5]])
s2 = np.array([[0.6, 0.0], [0.0, 1.1]])

m1 = np.array([0.0, 0.0])
m2 = np.array([2.0, -3.0])

X1 = np.random.multivariate_normal(mean=m1, cov=s1, size=200)
X2 = np.random.multivariate_normal(mean=m2, cov=s2, size=300)

X = np.vstack((X1, X2))

#np.savetxt("/tmp/test", X)

def test_gmm():
    gmm = pyclust.GMM(n_clusters=2)

    gmm.fit(X)
    print(gmm.priors_)
    print(gmm.means_)
    print(gmm.covars_)


    print(gmm.predict(X))

test_gmm()
