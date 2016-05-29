#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Latent Dirichlet Allocation + collapsed Gibbs sampling
# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.

import numpy as np
import random
import sys
from collections import defaultdict
import logging

def print_info(info_tuple):
    msg = '\t'.join(map(str, info_tuple))
    logging.info(msg)

class LDA:
    '''
    Latent Dirichlet Allocation with Collapsed Gibbs Sampling
    '''
    SAMPLING_RATE = 10
    def __init__(self, K, alpha, beta, docs, V):
        # set params
        self.K = K
        self.alpha = alpha
        self.beta = beta
        self.docs = docs
        self.V = V

        # init state
        self._init_state()

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
                p_z = self.n_z_w[:, t] * n_m_z / self.n_z
                new_z = np.random.multinomial(1, p_z / p_z.sum()).argmax()

                # set z the new topic and increment counters
                self.assignment(z_n, n_m_z, n, t, new_z)

    def discount(self, z_n, n_m_z, n, t):
        '''
        Cancel a topic assigned to a word slot
        '''
        z = z_n[n]
        n_m_z[z] -= 1
        self.n_z_w[z, t] -= 1
        self.n_z[z] -= 1

    def assignment(self, z_n, n_m_z, n, t, new_z):
        '''
        Assign a topic to a word slot
        '''
        z_n[n] = new_z
        n_m_z[new_z] += 1
        self.n_z_w[new_z, t] += 1
        self.n_z[new_z] += 1

    def worddist(self):
        '''
        phi = P(w|z): word probability of each topic
        '''
        return self.n_z_w / self.n_z[:, np.newaxis]

    def topicdist(self, docs=None):
        '''
        theta = P(z|d): topic probability of each document
        '''
        if docs == None:
            docs = self.docs
        doclens = np.array(list(map(len, docs)))
        return self.n_m_z / (doclens[:, np.newaxis] + self.K * self.alpha)

    def perplexity(self, docs=None):
        '''
        Compute the perplexity
        '''
        if docs == None:
            docs = self.docs
        phi = self.worddist()
        thetas = self.topicdist(docs)
        log_per = 0
        N = 0
        for m, doc in enumerate(docs):
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
        print_info(("PERP0", perp))
        for i in range(iteration):
            self.inference()
            if (i + 1) % self.SAMPLING_RATE == 0:
                perp = self.perplexity()
                print_info(("PERP%s" % (i+1), perp))
        self.output_word_dist_with_voca(voca)

    def word_dist_with_voca(self, voca, topk=10):
        '''
        Output the word probability of each topic
        '''
        zcount = np.zeros(self.K, dtype=int)
        wordcount = [defaultdict(int) for k in range(self.K)]
        for xlist, zlist in zip(self.docs, self.z_m_n):
            for x, z in zip(xlist, zlist):
                zcount[z] += 1
                wordcount[z][x] += 1

        phi = self.worddist()
        result = defaultdict(dict)
        for k in range(self.K):
            for w in np.argsort(-phi[k])[:topk]:
                result[k][voca[w]] = phi[k, w]
        return result

    def output_word_dist_with_voca(self, voca, topk=10):
        word_dist = self.word_dist_with_voca(voca, topk)
        for k in word_dist:
            for w in word_dist[k]:
                print_info(("TOPIC", k, w, word_dist[k][w]))

def main():
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    import dmr
    import argparse
    logging.basicConfig(level=logging.INFO, filename='lda.log')

    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str, help="corpus filename")
    parser.add_argument("--alpha", dest="alpha", type=float,
        help="parameter alpha", default=0.1)
    parser.add_argument("--beta", dest="beta", type=float,
        help="parameter beta", default=0.01)
    parser.add_argument("-k", dest="K", type=int,
        help="number of topics", default=50)
    parser.add_argument("-i", dest="iteration", type=int,
        help="iteration count", default=50)
    parser.add_argument("--df", dest="df", type=int,
        help="threshold of document freaquency to cut words", default=0)
    options = parser.parse_args()
    if not options.filename:
        parser.error("need corpus filename(-f)")

    corpus = dmr.Corpus.read(options.filename)
    voca = dmr.Vocabulary()
    docs = voca.read_corpus(corpus)
    if options.df > 0:
        docs = voca.cut_low_freq(docs, options.df)

    lda = dmr.LDA(options.K, options.alpha, options.beta, docs, voca.size())
    print_info(("BASIC", "corpus=%d, words=%d, K=%d, a=%f, b=%f, iter=%d"
        % (len(docs), len(voca.vocas), options.K,
        options.alpha, options.beta, options.iteration)))

    lda.learning(options.iteration, voca)

if __name__ == "__main__":
    main()
