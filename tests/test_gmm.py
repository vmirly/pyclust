import _gaussian_mixture_model as gmm
import numpy as np
import scipy, scipy.linalg



m1 = np.array([0,0])
m2 = np.array([3,3])

s1 = np.array([[1,0],[0,0.5]])
s2 = np.array([[0.1,0.4],[0.4,0.5]])

X = np.r_(np.random.multivariate_normal(m1, s1, size=100), \
          np.random.multivariate_normal(m2, s2, size=100) )



res = gmm._log_multivariate_density(X, list((m1,m2)), list((s1,s2)))


print(res[:10,:])


