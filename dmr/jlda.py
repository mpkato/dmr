#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
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
            # mu_z = kappa * mu + sum v_k
            # what should be added:
            # - kappa_z * (mu_z/kappa_z) (mu_z/kappa_z)^T
            self.sigma_z[k] += -np.outer(self.mu_z[k], self.mu_z[k])\
                / (self.kappa + self.n_z_v[k])
            self.sigma_z[k] += self.kappa * np.outer(self.mu, self.mu)

        x = np.array([ [[1,2],[3,4]] , [[1,2],[2,1]] ])
        #print(x.shape)
        #print(np.linalg.det(x).shape)
        #print(np.linalg.det(x))
        #print(pow(np.linalg.det(x), 2))
        #print(np.linalg.inv(x))
        #print(x[1][0] - x[0])
        #np.dot(np.dot((x - mu), np.linalg.inv(sigma)), (x - mu))

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

            for n, v in enumerate(vecs[m]):
                # discount for n-th word t with topic z
                self.vec_discount(z_v, n_m_z, n, v)

                # sampling topic new_z for t
                p_v = self.vector_probability(v)

                # set z the new topic and increment counters
                self.vec_assignment(z_v, n_m_z, n, v, new_z)

    def vec_discount(self, z_v, n_m_z, n, v):
        '''
        Cancel a topic assigned to a word slot
        '''
        z = z_v[n]
        n_m_z[z] -= 1

        if self.trained is None:
            self.sigma_z[z] -= -np.outer(self.mu_z[z], self.mu_z[z])\
                / (self.kappa + self.n_z_v[z])
            self.sigma_z[z] -= np.outer(v, v)
            self.mu_z[z] -= v
            self.n_z_v[z] -= 1
            self.sigma_z[z] += -np.outer(self.mu_z[z], self.mu_z[z])\
                / (self.kappa + self.n_z_v[z])

    def vec_assignment(self, z_v, n_m_z, n, v, new_z):
        '''
        Assign a topic to a word slot
        '''
        z_v[n] = new_z
        n_m_z[new_z] += 1

        if self.trained is None:
            self.sigma_z[new_z] -= -np.outer(self.mu_z[new_z], self.mu_z[new_z])\
                / (self.kappa + self.n_z_v[new_z])
            self.sigma_z[new_z] += np.outer(v, v)
            self.mu_z[new_z] += v
            self.n_z_v[new_z] += 1
            self.sigma_z[new_z] += -np.outer(self.mu_z[new_z], self.mu_z[new_z])\
                / (self.kappa + self.n_z_v[new_z])

    def vector_probability(self, v):
        kappa = self.n_z_v + self.kappa
        nu = self.n_z_v + self.nu
        mu = self.mu_z / kappa[:,np.newaxis]
        sigma = self.sigma_z\
            (((kappa + 1) / kappa) / (nu - len(self.vecs) + 1))[:,np.newaxis,np.newaxis]

    def simple_multivariate_t_distribution(x, mu, sigma, df, d):
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
        denom = pow(np.linalg.det(sigma), 1 / 2.0)\
            * pow(1 + (1. / df) * np.dot(np.dot((x - mu), np.linalg.inv(sigma)), (x - mu)), (d + df) / 2)
        result = 1.0 / denom
        return result

    def params(self):
        return ('''K=%d, alpha=%s, beta=%s, kappa=%s, nu=%s'''
            % (self.K, self.alpha, self.beta, self.kappa, self.nu))
