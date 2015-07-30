import numpy as np
import pyclust


d1 = np.random.uniform(low=0, high=2, size=(10,2))
d2 = np.random.uniform(low=3, high=5, size=(10,2))
d = np.vstack((d1,d2))

print(d)
print(d.shape)
kg = pyclust._kernel_kmeans._compute_gram_matrix(d, sigma_sq=2.0)

print(kg.shape)

print(kg[0,:])
print(kg[19,:])

w1 = np.random.randint(2, size=20)
w2 = 1 - w1
wmemb_old = np.vstack((w1,w2)).T


#print(wmemb_old.shape)
#print(wmemb_old)

for it in range(1):
  wmemb_new = np.zeros(shape=(20,2), dtype=int)
  kdist = np.empty(shape=(20,2), dtype=float)
  for i in range(20):
     kdist[i] = pyclust._kernel_kmeans._kernelized_dist2centers(kg, wmemb_old, i, 0)
     wmemb_new[i,np.argmin(kdist[i])] = 1

  #print(kdist)
  #print(wmemb_new)

  wmemb_old = wmemb_new.copy()


import sklearn
import sklearn.metrics.pairwise 

pk = sklearn.metrics.pairwise.pairwise_kernels(d, metric='rbf')
print(pk[0,:])
print(pk[19,:])
print(pk.shape)
