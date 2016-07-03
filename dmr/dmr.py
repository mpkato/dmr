#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.special as special
import scipy.optimize as optimize
import scipy.misc as misc
from .lda import LDA

class DMR(LDA):
    '''
    Topic Model with Dirichlet Multinomial Regression
    '''
    def __init__(self, K, sigma, beta, docs, vecs, V, trained=None):
        super(DMR, self).__init__(K, 0.0, beta, docs, V, trained)
        self.L = vecs.shape[1]
        self.vecs = vecs
        self.sigma = sigma
        self.Lambda = np.random.multivariate_normal(np.zeros(self.L),
            (self.sigma ** 2) * np.identity(self.L), size=self.K)
        if self.trained is None:
            self.alpha = self.get_alpha()
        else:
            self.alpha = self.get_alpha(self.trained.Lambda)

    def hyperparameter_learning(self):
        '''
        update alpha (overwrite)
        '''
        if self.trained is None:
            self.bfgs()
            self.alpha = self.get_alpha()

    def get_alpha_n_m_z(self, idx=None):
        if idx is None:
            return self.n_m_z + self.alpha
        else:
            return self.n_m_z[idx] + self.alpha[idx]

    def get_alpha(self, Lambda=None):
        '''
        alpha = exp(Lambda^T x)
        '''
        if Lambda is None:
            Lambda = self.Lambda
        return np.exp(np.dot(self.vecs, Lambda.T))

    def bfgs(self):
        def ll(x):
            x = x.reshape((self.K, self.L))
            return self._ll(x)

        def dll(x):
            x = x.reshape((self.K, self.L))
            result = self._dll(x)
            result = result.reshape(self.K * self.L)
            return result

        Lambda = np.random.multivariate_normal(np.zeros(self.L), 
            (self.sigma ** 2) * np.identity(self.L), size=self.K)
        Lambda = Lambda.reshape(self.K * self.L)

        newLambda, fmin, res = optimize.fmin_l_bfgs_b(ll, Lambda, dll)
        self.Lambda = newLambda.reshape((self.K, self.L))

    def _ll(self, x):
        result = 0.0
        # P(w|z)
        result += self.K * special.gammaln(self.beta * self.K)
        result += -np.sum(special.gammaln(np.sum(self.n_z_w, axis=1)))
        result += np.sum(special.gammaln(self.n_z_w))
        result += -self.V * special.gammaln(self.beta)

        # P(z|Lambda)
        alpha = self.get_alpha(x)
        result += np.sum(special.gammaln(np.sum(alpha, axis=1)))
        result += -np.sum(special.gammaln(
            np.sum(self.n_m_z+alpha, axis=1)))
        result += np.sum(special.gammaln(self.n_m_z+alpha))
        result += -np.sum(special.gammaln(alpha))

        # P(Lambda)
        result += -self.K / 2.0 * np.log(2.0 * np.pi * (self.sigma ** 2))
        result += -1.0 / (2.0 * (self.sigma ** 2)) * np.sum(x ** 2)

        result = -result
        return result

    def _dll(self, x):
        alpha = self.get_alpha(x)
        result = np.sum(self.vecs[:,np.newaxis,:] * alpha[:,:,np.newaxis]\
            * (special.digamma(np.sum(alpha, axis=1))[:,np.newaxis,np.newaxis]\
            - special.digamma(np.sum(self.n_m_z+alpha, axis=1))[:,np.newaxis,np.newaxis]\
            + special.digamma(self.n_m_z+alpha)[:,:,np.newaxis]\
            - special.digamma(alpha)[:,:,np.newaxis]), axis=0)\
            - x / (self.sigma ** 2)
        result = -result
        return result

    def params(self):
        return '''K=%d, sigma=%s, beta=%s''' % (self.K, self.sigma, self.beta)
