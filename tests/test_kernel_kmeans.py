import numpy as np
import pyclust


d1 = np.random.uniform(low=0, high=2, size=(100,2))
d2 = np.random.uniform(low=3, high=5, size=(100,2))
d = np.vstack((d1,d2))

from scipy.sparse import issparse


def test_kernelkmeans():
    kg = pyclust._kernel_kmeans._compute_gram_matrix(d, kern_type='rbf', sigma_sq=2.0)

    #print(kg[0,:])
    #print(kg[19,:])

    w = np.random.randint(2, size=20)
    wmemb_old = w

    kdist = np.zeros(shape=(20,2), dtype=float)

    print(pyclust._kernel_kmeans._fit_kernelkmeans(kg, 2, 100))
    return

    for it in range(10):

        kdist = np.empty(shape=(20,2), dtype=float)
        pyclust._kernel_kmeans._kernelized_dist2centers(kg, 2, wmemb_old, kdist)
        wmemb_new = np.argmin(kdist, axis=1)

        #print(kdist)
        #print(wmemb_new)

        wmemb_old = wmemb_new.copy()

    print(kdist)
    print(wmemb_new)

test_kernelkmeans()
