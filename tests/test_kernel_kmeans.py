import numpy as np
import pyclust


d1 = np.random.uniform(low=0, high=2, size=(10,2))
d2 = np.random.uniform(low=0, high=4, size=(10,2))
d = np.vstack((d1,d2))

print(d.shape)
kg = pyclust._kernel_kmeans._compute_gram_matrix(d, sigma_sq=2.0)

print(kg)
print(kg.shape)
