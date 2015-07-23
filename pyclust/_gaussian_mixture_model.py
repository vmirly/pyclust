
import warnings

import numpy as np
import scipy, scipy.linalg

from . import _kmeans



def _init_mixture_params(X, n_mixtures):
   """ 
      Initialize mixture density parameters with 
        equal priors
        random means
        identity covariance matrices
   """

   init_priors = np.ones(shape=n_mixtures, dtype=float) / n_mixtures

   init_means = _kmeans._kmeans_init(X, n_mixtures)
   
   n_features = X.shape[1]
   init_covars = np.empty(shape=(n_mixtures, n_features, n_features), dtype=float)
   for i in range(n_mixtures):
      init_covars[i] = np.eye(n_features)

   return(init_priors, init_means, init_covars)



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


def _log_multivariate_density(X, means, covars):
   """
      Class conditional density:
        P(x | mu, Sigma) = 1/((2pi)^d/2 * |Sigma|^1/2) * exp(-1/2 * (x-mu)^T * Sigma^-1 * (x-mu))

      log of class conditional density:
        log P(x | mu, Sigma) = -1/2*(d*log(2pi) + log(|Sigma|) + (x-mu)^T * Sigma^-1 * (x-mu))
   """
   n_samples, n_dim = X.shape
   n_components = means.shape[0]

   assert(means.shape[0] == covars.shape[0])

   log_proba = np.empty(shape=(n_samples, n_components), dtype=float)
   for i, (mu, cov) in enumerate(zip(means, covars)):
      cov_chol = scipy.linalg.cholesky(cov, lower=True)
      cov_log_det = 2 * np.sum(np.log(np.diagonal(cov_chol)))

      cov_solve = scipy.linalg.solve_triangular(cov_chol, (X - mu).T, lower=True).T

      log_proba[:, i] = - .5 * (np.sum(cov_solve ** 2, axis=1) + \
                       n_dim * np.log(2 * np.pi) + cov_log_det)
   return(log_proba)



def _log_likelihood_per_sample(X, means, covars):
   """
      Theta = (theta_1, theta_2, ... theta_M)
      Likelihood of mixture parameters given data: L(Theta | X) = product_i P(x_i | Theta)
      log likelihood: log L(Theta | X) = sum_i log(P(x_i | Theta))

      and note that p(x_i | Theta) = sum_j prior_j * p(x_i | theta_j)


      Probability of sample x being generated from component i:
         P(w_i | x) = P(x|w_i) * P(w_i) / P(X)
           where P(X) = sum_i P(x|w_i) * P(w_i)

       Here post_proba = P/(w_i | x)
        and log_likelihood = log(P(x|w_i))
   """
   logden = _log_multivariate_density(X, means, covars) 

   log_likelihood = np.log(np.sum(np.exp(logden), axis=1))
   post_proba = np.exp(logden - log_likelihood[:, np.newaxis])
   return (log_likelihood, post_proba)



def _maximization_step(X, posteriors):
   """ 
      Update class parameters as below:
        priors: P(w_i) = sum_x P(w_i | x) ==> Then normalize to get in [0,1]
        Class means: center_w_i = sum_x P(w_i|x)*x / sum_i sum_x P(w_i|x)
   """
   
   ### Prior probabilities or class weights
   sum_post_proba = np.sum(posteriors, axis=0)
   prior_proba = sum_post_proba / sum_post_proba.sum()
   
   ### means
   means = np.dot(posteriors.T, X) / sum_post_proba[:, np.newaxis]

   ### covariance matrices
   n_components = posteriors.shape[1]
   n_features = X.shape[1]
   covars = np.empty(shape=(n_components, n_features, n_features), dtype=float)
  
   for i in range(n_components):
      post_i = posteriors[:, i]
      mean_i = means[i]
      diff_i = X - mean_i
      covars[i] = np.dot(post_i * diff_i.T, diff_i) / post_i.sum()

   return(prior_proba, means, covars)



def _fit_gmm_params(X, n_mixtures, n_init, n_iter, tol):
   """
   """

   for init in range(n_init):
      priors, means, covars = _init_mixture_params(X, n_mixtures)

      for i in range(n_iter):
          ## E-step
          log_likelihoods, posteriors = _log_likelihood_per_sample(X, means, covars)

          ## M-step
          priors, means, covars = _maximization_step(X, posteriors)

   return(priors, means, covars)




class GMM(object):
   """ 
        Gaussian Mixture Model (GMM)

        Parameters
        -------


        Attibutes
        -------
           labels_   :  cluster labels for each data item
           

        Methods
        ------- 
           fit(X): fit the model
           fit_predict(X): fit the model and return the cluster labels
   """

   def __init__(self, n_clusters=2, n_trials=10, max_iter=100, tol=0.0001):
      assert n_clusters >= 2, 'n_clusters should be >= 2'
      self.n_clusters = n_clusters
      self.n_trials = n_trials
      self.max_iter = max_iter
      self.tol = tol
   
      self.converged = False

   def fit(self, X, y=None):
      """ Fit mixture-density parameters with EM algorithm
      """
      self.priors_, self.means_, self.covars_ = \
        _fit_gmm_params(X=X, n_mixtures=self.n_clusters, \
                        n_init=self.n_trials, n_iter=self.max_iter, \
                        tol=self.tol)

      self.converged = True


   def predict_proba(self, X):
      """
      """
      if not self.converged:
          raise Exception('Mixture model is not fit yet!! Try GMM.fit(X)')
      _, post_proba = _log_likelihood_per_sample(X=X, means=self.means_, covars=self.covars_)

      return(post_proba)


   def predict(self, X):
      """
      """
      post_proba = self.predict_proba(X)

      return(post_proba.argmax(axis=1))
