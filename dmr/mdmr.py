#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.special as special
import scipy.optimize as optimize
from .lda import print_info
from .dmr import DMR

class MDMR(DMR):
    '''
    Topic Model with Dirichlet Multinomial Regression with Multiple Data
    vecs: list(list(tuple(len, np.array)))
    '''
    def __init__(self, K, sigma, beta, docs, vecs, V, trained=None):
        super(DMR, self).__init__(K, 0.0, beta, docs, V, trained) # LDA
        self.L = vecs[0][0][1].shape[0]
        self.vecs = vecs
        self.sigma = sigma
        self.Lambda = np.random.multivariate_normal(np.zeros(self.L),
            (self.sigma ** 2) * np.identity(self.L), size=self.K)
        if self.trained is not None:
            alpha = self.get_alpha(self.trained.Lambda)
            self.n_m_z += alpha

    def get_alpha(self, Lambda):
        '''
        alpha = sum_l len_l * exp(Lambda^T x_l)
        '''
        alphas = []
        for lvecs in self.vecs:
            lens = np.array([l for l, _ in lvecs])
            vs = np.array([v for _, v in lvecs])
            alpha = np.sum(
                lens[:,np.newaxis] * np.exp(np.dot(vs, Lambda.T)), axis=0)
            alpha /= np.sum(lens)
            alphas.append(alpha)
        alphas = np.array(alphas)
        return alphas

    def _dll(self, x):
        alpha = self.get_alpha(x)
        result = np.sum(self._dll_common(x)\
            * (special.digamma(np.sum(alpha, axis=1))[:,np.newaxis,np.newaxis]\
            - special.digamma(np.sum(self.n_m_z+alpha, axis=1))[:,np.newaxis,np.newaxis]\
            + special.digamma(self.n_m_z+alpha)[:,:,np.newaxis]\
            - special.digamma(alpha)[:,:,np.newaxis]), axis=0)\
            - x / (self.sigma ** 2)
        result = -result
        return result

    def _dll_common(self, x):
        '''
        sum_l len_l * x_{m_l_c} * exp(Lambda^T x_{m_l})
        '''
        alphas = []
        for m in self.vecs:
            alpha = []
            vs = []
            lens = []
            for l, v in m:
                alpha.append(l * np.exp(np.dot(v, x.T)))
                lens.append(l)
                vs.append(v)
            alpha = np.array(alpha)
            alpha /= np.sum(lens) # l x k
            vs = np.array(vs) # l x c
            res = np.dot(alpha.T, vs) # k x c
            alphas.append(res)
        result = np.array(alphas) # d x k x c
        return result
