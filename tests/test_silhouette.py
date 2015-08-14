import numpy as np
import scipy, scipy.linalg
import sys

import pyclust, pyclust.validate
from sklearn.metrics import silhouette_score, silhouette_samples

s1 = np.array([[0.3, 0.2], [0.7, 0.5]])
s2 = np.array([[0.6, 0.0], [0.0, 1.1]])

m1 = np.array([0.0, 0.0])
m2 = np.array([2.0, -3.0])

X1 = np.random.multivariate_normal(mean=m1, cov=s1, size=200)
X2 = np.random.multivariate_normal(mean=m2, cov=s2, size=300)

X = np.vstack((X1, X2))

#X = np.array([[-1,0],[0,0],[-1,-1],[2,0],[2,1],[3,0]])
#X = np.array([[-1,0],[0,0],[0,0],[2,0],[2,0],[3,0]])
#ypred = np.array([1,1,1,2,2,2])
ypred = np.hstack((np.zeros(shape=200), np.ones(shape=300)))
print(ypred.shape)

#np.savetxt("/tmp/test", X)

def test_gmm():
    sil = pyclust.validate.Silhouette()
    sil_score = sil.score(X, ypred, sample_size=None)

    print(sil_score[0])

    print(sil.sample_scores[:10])

    print(silhouette_score(X, ypred, sample_size=None))
    
    print(silhouette_samples(X, ypred)[:10])
test_gmm()
