import numpy as np
import scipy, scipy.linalg
import sys

sys.path.append("pyclust/")
import _gaussian_mixture_model as gmm


m1 = np.array([0,0])
m2 = np.array([3,3])

s1 = np.matrix([[1,0],[0,0.5]])
s2 = np.matrix([[1,0.0],[0.0,0.5]])

X = np.vstack((np.random.multivariate_normal(m1, s1, size=100), \
               np.random.multivariate_normal(m2, s2, size=100)) )

print(X.shape)

res = gmm._log_multivariate_density(X, list((m1, m1, m2)), list((s1, s2, s2)))

print(res[:10,:])


print(gmm.__log_density_single(X[0,:], m1, s1), \
      gmm.__log_density_single(X[0,:], m1, s2), \
      gmm.__log_density_single(X[0,:], m2, s2))


log1, _ = gmm._log_likelihood_per_sample(X, list((m1, m1, m2)), list((s1, s2, s2)))
from sklearn.utils.extmath import logsumexp
log2 = logsumexp(res, axis=1)

w = _.sum(axis=0)
print(w)
weighted_X_sum = np.dot(_.T, X)
print(weighted_X_sum)

print(np.any(np.abs(log1 - log2) < 0.001))

priors, means = gmm._maximization_step(X, _)

print(priors, priors.sum())
print(means)
