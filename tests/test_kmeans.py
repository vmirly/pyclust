import numpy as np
import pyclust


d1 = np.random.uniform(low=0, high=2, size=(10,2))
d2 = np.random.uniform(low=0, high=4, size=(10,2))
d = np.vstack((d1,d2))

print(d.shape)
cent = pyclust._kmeans._kmeans_init(d, 2)
#d2c = scipy.spatial.distance.cdist(d, cent, metric='euclidean')

print(cent)

membs = pyclust._kmeans._assign_clusters(d, cent)

print(membs)

cent_upd = pyclust._kmeans._update_centers(d, membs, n_clusters=2)

print(cent_upd)

print(pyclust._kmeans._kmeans_run(d, n_clusters=2, max_iter=20, tol=0.0001))

km = pyclust.KMeans(n_clusters=2)

km.fit(d)

print("Centers: ", km.centers_)
print("Labels: ", km.labels_)
print("SSE: ", km.sse_arr_)
print("N_ITER: ", km.n_iter_)

## Checking the random-number generator
#rng = np.random.RandomState(1234)
print('\n\n*** Testing RandomState: ***')
for i in range(3):
   rng = np.random.RandomState(1234)
   print(pyclust._kmeans._kmeans_init(d, 2, rng=rng))
   print()
