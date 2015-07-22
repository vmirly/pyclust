
import warnings

import numpy as np
import scipy, scipy.linalg


def __log_density_single(x, mean, covar):
   """ This is just a test function to calculate 
       the normal density at x given mean and covariance matrix.

       Note: this function is not efficient, so
             _log_multivariate_density is recommended for use.
   """
   n_dim = mean.shape[0]

   dx = x - mean
   covar_inv = scipy.linalg.inv(covar)
   covar_det = scipy.linalg.det(covar)

   #print(covar_inv)
   #print(covar_det)
   den = np.dot(np.dot(dx.T, covar_inv), dx) + n_dim*np.log(2*np.pi) + np.log(covar_det)

   return(-1/2 * den)


def _log_multivariate_density(X, mean_list, covar_list):
   """ 
   """
   print(type(mean_list), type(covar_list), mean_list, covar_list)
   n_samples, n_dim = X.shape
   n_mix = len(mean_list)

   print(n_mix)

   log_proba = np.empty(shape=(n_samples, n_mix), dtype=float)
   for i, (mu, cov) in enumerate(zip(mean_list, covar_list)):
      print(i, mu, cov)
      cov_chol = scipy.linalg.cholesky(cov, lower=True)
      cov_log_det = 2 * np.sum(np.log(np.diagonal(cov_chol)))

      cov_solve = scipy.linalg.solve_triangular(cov_chol, (X - mu).T, lower=True).T

      log_proba[:, i] = - .5 * (np.sum(cov_solve ** 2, axis=1) + \
                       n_dim * np.log(2 * np.pi) + cov_log_det)
   return(log_proba)

def _fit_gmm(X, n_components):
   """
   """
   # Initialize the means using KMeans
   means = _kmeans.KMeans(n_clusters=n_components).fit(X).centers_
   # Initialize the covariance matrix
   cov_all = np.cov(X.T)



def _log_likelihood_per_sample(X, mean_list, covar_list):
   """
      P(w_i | x) = P(x|w_i) * P(w_i) / P(X)
       where P(X) = sum_i P(x|w_i) * P(w_i)

       Here post_proba = P/(w_i | x)
        and log_likelihood = log(P(x|w_i))
   """
   logden = _log_multivariate_density(X, mean_list, covar_list) 

   log_likelihood = np.log(np.sum(np.exp(logden), axis=1))
   post_proba = np.exp(logden - log_likelihood[:, np.newaxis])
   return (log_likelihood, post_proba)


def _maximization_step(X, posteriors):
   """
   """
   weights = np.sum(posteriors, axis=0)
   
