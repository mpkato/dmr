#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.special as special
import scipy.optimize as optimize
from .lda import print_info
from .dmr import DMR

class CDMR(DMR):
    '''
    Topic Model with Coupled Dirichlet Multinomial Regression
    '''
    def __init__(self, K, sigma, beta, docs, vecs, V):
        super(CDMR, self).__init__(K, sigma, beta, docs, vecs, V)
        if self.K % 2 != 0:
            raise Exception("K must be odd in CDMR")

    def bfgs(self):
        def ll(x):
            # the dimension is a half of K
            x = x.reshape((self.K//2, self.L))
            x = np.vstack((x, -x))
            return self._ll(x)

        def dll(x):
            # the dimension is a half of K
            x = x.reshape((self.K//2, self.L))
            result = self._dll(x)
            result = result.reshape(self.K * self.L // 2)
            return result

        # the dimension is a half of K
        Lambda = np.random.multivariate_normal(np.zeros(self.L), 
            (self.sigma ** 2) * np.identity(self.L), size=self.K//2)
        Lambda = Lambda.reshape(self.K * self.L // 2)

        newLambda, fmin, res = optimize.fmin_l_bfgs_b(ll, Lambda, dll)

        # the dimension is a half of K
        newLambda = newLambda.reshape((self.K//2, self.L))
        self.Lambda = np.vstack((newLambda, -newLambda))

    def _dll(self, x):
        palpha = np.exp(np.dot(self.vecs, x.T))
        nalpha = np.exp(-np.dot(self.vecs, x.T))
        alpha = np.hstack((palpha, nalpha))
        result = np.sum(self.vecs[:,np.newaxis,:]\
            * ((palpha[:,:,np.newaxis] - nalpha[:,:,np.newaxis])\
            * (special.digamma(np.sum(alpha, axis=1))\
            - special.digamma(np.sum(self.n_m_z+alpha, axis=1))
            )[:,np.newaxis,np.newaxis]), axis=0)
        result += np.sum(self.vecs[:,np.newaxis,:]\
            * (palpha * (special.digamma(self.n_m_z[:,:self.K//2]+palpha)\
                - special.digamma(palpha))\
                - nalpha * (special.digamma(self.n_m_z[:,self.K//2:]+nalpha)\
                - special.digamma(nalpha))
                )[:,:,np.newaxis], axis=0)
        result += - x / (self.sigma ** 2)
        result = -result
        return result
