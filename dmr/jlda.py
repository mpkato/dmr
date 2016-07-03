#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from choldate import cholupdate, choldowndate
import scipy.special as special
import scipy.constants as constants
import scipy.optimize as optimize
from .lda import LDA

class JLDA(LDA):
    '''
    Join Latent Dirichlet Allocation
    '''
    def __init__(self, K, alpha, beta, kappa, nu, docs, vecs, V, trained=None):
        super(JLDA, self).__init__(K, alpha, beta, docs, V, trained)
        self.L = vecs[0][0][1].shape[0]
        self.vecs = vecs
        self.kappa = kappa
        self.nu = nu

        self.mu = np.zeros(self.L) # for simplicity
        self.Phi = np.identity(self.L) # for simplicity
        self._init_state_gaussian()

    def _init_state_gaussian(self):
        self.mu_z = np.zeros((self.K, self.L))\
            + self.kappa * self.mu[np.newaxis, :]
        self.sigma_z = np.zeros((self.K, self.L, self.L))\
            + self.Phi[np.newaxis, :, :]

        self.z_m_v = []
        self.n_z_v = np.zeros(self.K)

        self.N_v = 0
        for m, vec in enumerate(self.vecs):
            self.N_v += len(vec)
            z_n = []
            for l, v in vec:
                z = np.random.randint(0, self.K)
                z_n.append(z)
                self.n_m_z[m, z] += 1
                self.mu_z[z] += v
                self.sigma_z[z] += np.outer(v, v) # v v^T
                self.n_z_v[z] += 1
            self.z_m_v.append(np.array(z_n))

        for k in range(self.K):
            # kappa_z * mu_z  mu_z^T
            # = kappa_z * (mu_z/kappa_z) (mu_z/kappa_z)^T
            self.sigma_z[k] += -np.outer(self.mu_z[k], self.mu_z[k])\
                / (self.kappa + self.n_z_v[k])
            # kappa * mu mu^T
            self.sigma_z[k] += self.kappa * np.outer(self.mu, self.mu)

    def inference(self):
        '''
        Re-assignment of topics to words
        '''
        for m, doc in enumerate(self.docs):
            z_n = self.z_m_n[m]
            z_v = self.z_m_v[m]
            n_m_z = self.n_m_z[m]
            for n, t in enumerate(doc):
                # discount for n-th word t with topic z
                self.discount(z_n, n_m_z, n, t)

                # sampling topic new_z for t
                p_z = self.n_z_w[:, t] * n_m_z / self.n_z
                new_z = np.random.multinomial(1, p_z / p_z.sum()).argmax()

                # set z the new topic and increment counters
                self.assignment(z_n, n_m_z, n, t, new_z)

            for n, (l, v) in enumerate(self.vecs[m]):
                # discount for n-th word t with topic z
                self.vec_discount(z_v, n_m_z, n, v)

                # sampling topic new_z for t
                p_v = self.vector_probability(v)
                new_z = np.random.multinomial(1, p_v / p_v.sum()).argmax()

                # set z the new topic and increment counters
                self.vec_assignment(z_v, n_m_z, n, v, new_z)

    def vec_discount(self, z_v, n_m_z, n, v):
        '''
        Cancel a topic assigned to a word slot
        '''
        z = z_v[n]
        n_m_z[z] -= 1

        if self.trained is None:
            kappa_z = self.kappa + self.n_z_v[z]
            x = self.mu_z[z] / kappa_z - v
            self.sigma_z[z] -= (kappa_z / (kappa_z - 1)) * np.outer(x, x)
            self.mu_z[z] -= v
            self.n_z_v[z] -= 1

    def vec_assignment(self, z_v, n_m_z, n, v, new_z):
        '''
        Assign a topic to a word slot
        '''
        z_v[n] = new_z
        n_m_z[new_z] += 1

        if self.trained is None:
            self.mu_z[new_z] += v
            self.n_z_v[new_z] += 1
            kappa_z = self.kappa + self.n_z_v[new_z]
            x = self.mu_z[new_z] / kappa_z - v
            self.sigma_z[new_z] += (kappa_z / (kappa_z - 1)) * np.outer(x, x)

    def vector_probability(self, v):
        kappa = self.n_z_v + self.kappa
        nu = self.n_z_v + self.nu
        mu = self.mu_z / kappa[:,np.newaxis]
        df = nu - self.L + 1
        sigma = self.sigma_z * ((kappa + 1) / kappa / df)[:,np.newaxis,np.newaxis]
        dim = self.L
        result = self.simple_multivariate_t_distribution(v, mu, sigma, df, dim)
        return result

    def simple_multivariate_t_distribution(self, x, mu, sigma, df, d):
        '''
        Multivariate t-student density:
        output:
            the density of the given element
        input:
            x = parameter (d dimensional numpy array or scalar)
            mu = mean (d dimensional numpy array or scalar)
            Sigma = scale matrix (dxd numpy array)
            df = degrees of freedom
            d: dimension
        '''
        num = special.gamma((df + d)/2)
        xSigma = np.dot((x - mu), np.linalg.inv(sigma))
        xSigma = np.array([xSigma[i, i] for i in range(self.K)])
        denom = special.gamma(df / 2) * np.power(df * np.pi, d / 2.0)\
            * np.power(np.linalg.det(sigma), 1 / 2.0)\
            * np.power(1 + (1. / df)\
            * np.sum(xSigma * (x - mu), axis=1), (d + df) / 2)
        result = num / denom
        return result

    def params(self):
        return ('''K=%d, alpha=%s, beta=%s, kappa=%s, nu=%s'''
            % (self.K, self.alpha, self.beta, self.kappa, self.nu))
