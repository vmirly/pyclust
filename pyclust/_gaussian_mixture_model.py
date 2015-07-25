
import warnings

import numpy as np
import scipy, scipy.linalg
import sklearn, sklearn.utils
from . import _kmeans

Epsilon = 100 * np.finfo(float).eps
Lambda = 0.1

def _init_mixture_params(X, n_mixtures):
   """ 
      Initialize mixture density parameters with 
        equal priors
        random means
        identity covariance matrices
   """

   init_priors = np.ones(shape=n_mixtures, dtype=float) / n_mixtures

   km = _kmeans.KMeans(n_clusters = n_mixtures)
   km.fit(X)
   init_means = km.centers_ #_kmeans_init(X, n_mixtures)
   
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
      try:
          cov_chol = scipy.linalg.cholesky(cov, lower=True)
      except: # scipy.linalg.LinAlgError:
          print(cov)
          raise ValueError("Triangular Matrix")

      cov_log_det = 2 * np.sum(np.log(np.diagonal(cov_chol)))

      cov_solve = scipy.linalg.solve_triangular(cov_chol, (X - mu).T, lower=True).T

      log_proba[:, i] = -0.5 * (np.sum(cov_solve ** 2, axis=1) + \
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

   logden_max = logden.max(axis=1)
   log_likelihood = np.log(np.sum(np.exp(logden.T - logden_max) + Epsilon, axis=0))
   log_likelihood += logden_max

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
   prior_proba = sum_post_proba / (sum_post_proba.sum() + Epsilon)
   
   ### means
   means = np.dot(posteriors.T, X) / (sum_post_proba[:, np.newaxis] + Epsilon)

   ### covariance matrices
   n_components = posteriors.shape[1]
   n_features = X.shape[1]
   covars = np.empty(shape=(n_components, n_features, n_features), dtype=float)
  
   for i in range(n_components):
      post_i = posteriors[:, i]
      mean_i = means[i]
      diff_i = X - mean_i

      with np.errstate(under='ignore'):
          covar_i = np.dot(post_i * diff_i.T, diff_i) / (post_i.sum() + Epsilon)
      covars[i] = covar_i + Lambda * np.eye(n_features)

      if (not np.allclose(covars[i], covars[i].T) or np.any(scipy.linalg.eigvalsh(covars[i]) <= 0)):
         raise ValueError("Component %d of covars must be positive-definite" % i)

   return(prior_proba, means, covars)



def _fit_gmm_params(X, n_mixtures, n_init, n_iter, tol):
   """
   """

   best_mean_loglikelihood = -np.infty

   for init in range(n_init):
      priors, means, covars = _init_mixture_params(X, n_mixtures)

      prev_mean_loglikelihood = None
      for i in range(n_iter):
          ## E-step
          log_likelihoods, posteriors = _log_likelihood_per_sample(X, means, covars)

          ## M-step
          priors, means, covars = _maximization_step(X, posteriors)

          ## convergence Check
          curr_mean_loglikelihood = log_likelihoods.mean()

          if prev_mean_loglikelihood is not None:
              if np.abs(curr_mean_loglikelihood - prev_mean_loglikelihood) < tol:
                  break

          prev_mean_loglikelihood = curr_mean_loglikelihood

      if curr_mean_loglikelihood > best_mean_loglikelihood:
          best_mean_loglikelihood = curr_mean_loglikelihood
          best_params = {
                  'priors' : priors,
                  'means'  : means,
                  'covars' : covars,
                  'mean_log_likelihood' : curr_mean_loglikelihood,
                  'n_iter' : i
                }

   return(best_params)




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
      params_dict = _fit_gmm_params(X=X, n_mixtures=self.n_clusters, \
                        n_init=self.n_trials, n_iter=self.max_iter, \
                        tol=self.tol)
      self.priors_ = params_dict['priors']
      self.means_  = params_dict['means']
      self.covars_ = params_dict['covars']

      self.converged = True
      self.labels_ = self.predict(X)


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
