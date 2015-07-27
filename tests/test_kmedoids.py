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

cent_upd = pyclust._kmedoids._update_centers(d, membs, n_clusters=2, distance='euclidean')

print(cent_upd)

print(pyclust._kmedoids._kmedoids_run(d, n_clusters=2, distance='euclidean', max_iter=20, tol=0.001))

kmd = pyclust.KMedoids(n_clusters=2)

kmd.fit(d)

print("Centers: ", kmd.centers_)
print("Labels: ", kmd.labels_)
print("SSE: ", kmd.sse_arr_)
print("N_ITER: ", kmd.n_iter_)
