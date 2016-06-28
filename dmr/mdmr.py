#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.special as special
import scipy.optimize as optimize
from .dmr import DMR

class MDMR(DMR):
    '''
    Topic Model with Dirichlet Multinomial Regression with Multiple Data
    vecs: list(list(tuple(len, np.array)))
    '''
    def __init__(self, K, sigma, beta, docs, vecs, V, trained=None):
        super(DMR, self).__init__(K, 0.0, beta, docs, V, trained) # LDA
        self.L = vecs[0][0][1].shape[0]
        self._set_lens_vecs(vecs)
        self.sigma = sigma
        self.Lambda = np.random.multivariate_normal(np.zeros(self.L),
            (self.sigma ** 2) * np.identity(self.L), size=self.K)
        self.prev_alpha = 0.0
        if self.trained is not None:
            alpha = self.get_alpha(self.trained.Lambda)
            self.n_m_z += alpha

    def _set_lens_vecs(self, vecs):
        self.lens = [np.array([l for l, _ in vs]) for vs in vecs]
        self.lens = [lv / np.sum(lv) for lv in self.lens]
        self.vecs = [np.array([v for _, v in vs]) for vs in vecs]

    def get_alpha(self, Lambda):
        '''
        alpha = sum_l len_l * exp(Lambda^T x_l)
        '''
        return np.array([np.sum(
            ls[:,np.newaxis] * np.exp(np.dot(vs, Lambda.T)), axis=0)
                for ls, vs in zip(self.lens, self.vecs)])

    def _dll(self, x):
        alpha = self.get_alpha(x)
        return -(np.sum(self._dll_common(x)\
            * (special.digamma(np.sum(alpha, axis=1))[:,np.newaxis,np.newaxis]\
            - special.digamma(np.sum(self.n_m_z+alpha, axis=1))[:,np.newaxis,np.newaxis]\
            + special.digamma(self.n_m_z+alpha)[:,:,np.newaxis]\
            - special.digamma(alpha)[:,:,np.newaxis]), axis=0)\
            - x / (self.sigma ** 2))

    def _dll_common(self, x):
        '''
        sum_l len_l * x_{m_l_c} * exp(Lambda^T x_{m_l})
        '''
        return np.array([
            np.dot((ls[:,np.newaxis] * np.exp(np.dot(vs, x.T))).T, vs)
            for ls, vs in zip(self.lens, self.vecs)])
