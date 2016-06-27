#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.special as special
import scipy.optimize as optimize
from .lda import print_info, LDA

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
        if self.trained is not None:
            alpha = self.get_alpha(self.trained.Lambda)
            self.n_m_z += alpha

    def learning(self, iteration, voca):
        '''
        Repeat inference for learning with alpha update
        '''
        perp = self.perplexity()
        print_info(("PERP0", perp))
        alpha = 0.0
        for i in range(iteration):

            # update alpha
            if self.trained is None:
                self.n_m_z -= alpha
                self.bfgs()
                alpha = self.get_alpha(self.Lambda)
                self.n_m_z += alpha

            self.inference()
            if (i + 1) % self.SAMPLING_RATE == 0:
                perp = self.perplexity()
                print_info(("PERP%s" % (i+1), perp))
        self.output_word_dist_with_voca(voca)

    def get_alpha(self, Lambda):
        '''
        alpha = exp(Lambda^T x)
        '''
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

    def perplexity(self):
        '''
        Compute the perplexity
        '''
        if self.trained is None:
            Lambda = self.Lambda
        else:
            Lambda = self.trained.Lambda
        alpha = self.get_alpha(Lambda)

        Kalpha = np.sum(alpha, axis=1)
        phi = self.worddist()
        log_per = 0
        N = 0
        for m, doc in enumerate(self.docs):
            theta = self.n_m_z[m] / (len(self.docs[m]) + Kalpha[m])
            for w in doc:
                log_per -= np.log(np.inner(phi[:,w], theta))
            N += len(doc)
        return np.exp(log_per / N)

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
