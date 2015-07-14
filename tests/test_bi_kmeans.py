import numpy as np
import pyclust


d1 = np.random.uniform(low=0, high=2, size=(10,2))
d2 = np.random.uniform(low=0, high=4, size=(10,2))
d = np.vstack((d1,d2))

print(d.shape)


bkm = pyclust.BisectKMeans(n_clusters=3)

bkm.fit(d)


print(bkm.labels_)

leaf_nodes = bkm.tree_.all_nodes()
print(leaf_nodes)
for node in leaf_nodes:
   print(node.data)
