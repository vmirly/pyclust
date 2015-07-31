import numpy as np
import pyclust



d1 = np.random.uniform(low=0, high=2, size=(3,2))
d2 = np.random.uniform(low=3, high=5, size=(3,2))
d = np.vstack((d1,d2))

from scipy.sparse import issparse




def test_kernelkmeans():
    kg = pyclust._kernel_kmeans._compute_gram_matrix(d, kern_type='rbf', sigma_sq=2.0)

    print(kg[0,:])
    #print(kg[19,:])

    from sklearn.metrics.pairwise import pairwise_kernels

    params = {"gamma": 0.5}
              #"degree": 3,
              #"coef0": 1}
    kp = pairwise_kernels(d, metric='rbf', **params)

    print(kp[0,:])


    return


    w1 = np.random.randint(2, size=20)
    w2 = 1 - w1
    wmemb_old = np.vstack((w1,w2)).T


    #print(wmemb_old.shape)
    #print(wmemb_old)

    for it in range(1):
        wmemb_new = np.zeros(shape=(20,2), dtype=int)
        kdist = np.empty(shape=(20,2), dtype=float)
        for i in range(20):
            kdist[i] = pyclust._kernel_kmeans._kernelized_dist2centers(kg, wmemb_old, i)
            wmemb_new[i,np.argmin(kdist[i])] = 1

        #print(kdist)
        #print(wmemb_new)

        wmemb_old = wmemb_new.copy()



test_kernelkmeans()
