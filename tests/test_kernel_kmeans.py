import numpy as np
import pyclust


d1 = np.random.uniform(low=0, high=3, size=(20,2))
d2 = np.random.uniform(low=3, high=5, size=(20,2))
d = np.vstack((d1,d2))
from scipy.sparse import issparse

def test_kernelkmeans():
    kg = pyclust._kernel_kmeans._compute_gram_matrix(d, kernel_type='rbf', params={})

    #print(kg[0,:])
    #print(kg[19,:])

    w = np.random.randint(2, size=20)
    wmemb_old = w

    kdist = np.zeros(shape=(20,2), dtype=float)

    #res = pyclust._kernel_kmeans._fit_kernelkmeans(kg, 2, 100)

    kkm = pyclust.KernelKMeans(n_clusters=2, kernel='rbf')
    kkm.fit_predict(d)
    print("Converged after %d iterations"%kkm.n_iter_)
    print(kkm.labels_)

test_kernelkmeans()
