#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latent Dirichlet Allocation + collapsed Gibbs sampling
# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.

import numpy as np
import random
import sys
from collections import defaultdict
from logging import getLogger

class LDA:
    '''
    Latent Dirichlet Allocation with Collapsed Gibbs Sampling
    '''
    SAMPLING_RATE = 10
    def __init__(self, K, alpha, beta, docs, V, trained=None):
        # set params
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.docs = docs
        self.V = V

        # init state
        self._init_state()
        self.trained = trained
        if self.trained is not None:
            self.n_z_w = self.trained.n_z_w
            self.n_z = self.trained.n_z

        # init logger
        self.logger = getLogger(self.__class__.__name__)

    def _init_state(self):
        '''
        Initialize
            - z_m_n: topics assigned to word slots in documents
            - n_m_z: freq. of topics assigned to documents
            - n_z_w: freq. of words assigned to topics
            - n_z:   freq. of topics assigned
        '''
        # assign zero + hyper-params
        self.z_m_n = []
        self.n_m_z = np.zeros((len(self.docs), self.K)) + self.alpha
        self.n_z_w = np.zeros((self.K, self.V)) + self.beta
        self.n_z = np.zeros(self.K) + self.V * self.beta

        # randomly assign topics
        self.N = 0
        for m, doc in enumerate(self.docs):
            self.N += len(doc)
            z_n = []
            for t in doc:
                z = np.random.randint(0, self.K)
                z_n.append(z)
                self.n_m_z[m, z] += 1
                self.n_z_w[z, t] += 1
                self.n_z[z] += 1
            self.z_m_n.append(np.array(z_n))

    def get_alpha_n_m_z(self, idx=None):
        '''
        Return self.n_m_z (including alpha)
        '''
        if idx is None:
            return self.n_m_z
        else:
            return self.n_m_z[idx]

    def inference(self):
        '''
        Re-assignment of topics to words
        '''
        for m, doc in enumerate(self.docs):
            z_n = self.z_m_n[m]
            n_m_z = self.n_m_z[m]
            for n, t in enumerate(doc):
                # discount for n-th word t with topic z
                self.discount(z_n, n_m_z, n, t)

                # sampling topic new_z for t
                p_z = self.n_z_w[:, t] * self.get_alpha_n_m_z(m) / self.n_z
                try:
                    new_z = np.random.multinomial(1, p_z / p_z.sum()).argmax()
                except Exception as e:
                    print(self.get_alpha(self.Lambda)[m])
                    print(p_z)
                    print(p_z / np.sum(p_z))
                    print(np.sum(p_z / np.sum(p_z)))
                    raise e

                # set z the new topic and increment counters
                self.assignment(z_n, n_m_z, n, t, new_z)

    def discount(self, z_n, n_m_z, n, t):
        '''
        Cancel a topic assigned to a word slot
        '''
        z = z_n[n]
        n_m_z[z] -= 1

        if self.trained is None:
            self.n_z_w[z, t] -= 1
            self.n_z[z] -= 1

    def assignment(self, z_n, n_m_z, n, t, new_z):
        '''
        Assign a topic to a word slot
        '''
        z_n[n] = new_z
        n_m_z[new_z] += 1

        if self.trained is None:
            self.n_z_w[new_z, t] += 1
            self.n_z[new_z] += 1

    def worddist(self):
        '''
        phi = P(w|z): word probability of each topic
        '''
        return self.n_z_w / self.n_z[:, np.newaxis]

    def get_alpha(self):
        '''
        fixed alpha
        '''
        return self.alpha

    def topicdist(self):
        '''
        theta = P(z|d): topic probability of each document
        '''
        doclens = np.array(list(map(len, self.docs)))
        return self.get_alpha_n_m_z()\
            / (doclens[:, np.newaxis] + self.K * self.get_alpha())

    def perplexity(self):
        '''
        Compute the perplexity
        '''
        if self.trained is None:
            phi = self.worddist()
        else:
            phi = self.trained.worddist()
        thetas = self.topicdist()
        log_per = 0
        N = 0
        for m, doc in enumerate(self.docs):
            theta = thetas[m]
            for w in doc:
                log_per -= np.log(np.inner(phi[:,w], theta))
            N += len(doc)
        return np.exp(log_per / N)

    def learning(self, iteration, voca):
        '''
        Repeat inference for learning
        '''
        perp = self.perplexity()
        self.log(self.logger.info, "PERP0", [perp])
        for i in range(iteration):
            self.hyperparameter_learning()
            self.inference()
            if (i + 1) % self.SAMPLING_RATE == 0:
                perp = self.perplexity()
                self.log(self.logger.info, "PERP%s" % (i+1), [perp])
        self.output_word_dist_with_voca(voca)

    def hyperparameter_learning(self):
        '''
        No hyperparameter learning in LDA
        '''
        pass

    def word_dist_with_voca(self, voca, topk=None):
        '''
        Output the word probability of each topic
        '''
        phi = self.worddist()
        if topk is None:
            topk = phi.shape[1]
        result = defaultdict(dict)
        for k in range(self.K):
            for w in np.argsort(-phi[k])[:topk]:
                result[k][voca[w]] = phi[k, w]
        return result

    def output_word_dist_with_voca(self, voca, topk=10):
        word_dist = self.word_dist_with_voca(voca, topk)
        for k in word_dist:
            word_dist[k] = sorted(word_dist[k].items(),
                key=lambda x: x[1], reverse=True)
            for w, v in word_dist[k]:
                self.log(self.logger.debug, "TOPIC", [k, w, v])

    def log(self, method, etype, messages):
        method("\t".join(map(str, [self.params(), etype] + messages)))

    def params(self):
        return '''K=%d, alpha=%s, beta=%s''' % (self.K, self.alpha, self.beta)

    def __getstate__(self):
        '''
        logger cannot be serialized
        '''
        state = self.__dict__.copy()
        del state['logger']
        return state

    def __setstate__(self, state):
        '''
        logger cannot be serialized
        '''
        self.__dict__.update(state)
        self.logger = getLogger(self.__class__.__name__)
