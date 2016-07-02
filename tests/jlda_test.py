# -*- coding:utf-8 -*-
import unittest
import nose
import dmr
import os
import numpy as np
from tests.settings import (DMR_DOC_FILEPATH, DMR_VEC_FILEPATH,
    K, ALPHA, BETA, KAPPA, NU, L, mk_dmr_dat, count_word_freq)

class JLDATestCase(unittest.TestCase):
    NUM_VECS = 10
    def setUp(self):
        np.random.seed(0)
        if not os.path.exists(DMR_DOC_FILEPATH)\
            or not os.path.exists(DMR_VEC_FILEPATH):
            mk_dmr_dat()

    def _init_jlda(self):
        corpus = dmr.Corpus.read(DMR_DOC_FILEPATH)
        vcorpus = dmr.Corpus.read(DMR_VEC_FILEPATH, dtype=float)

        # General 1-NUM_VECS vecs per doc with mean=vec
        vecs = [[(1.0, np.random.normal(loc=vec))
            for i in range(np.random.randint(1, self.NUM_VECS))]
                for vec in vcorpus]

        voca = dmr.Vocabulary()
        docs = voca.read_corpus(corpus)
        lda = dmr.JLDA(K, ALPHA, BETA, KAPPA, NU, docs, vecs, voca.size())
        return voca, docs, vecs, lda

    def test_jlda___init__(self):
        '''
        JLDA.__init__
        '''
        voca, docs, vecs, lda = self._init_jlda()

        # L
        self.assertEqual(lda.L, L)

        # n_m_z
        self.assertAlmostEqual(np.sum(lda.n_m_z[0]-ALPHA), 10+len(vecs[0]))
        self.assertAlmostEqual(np.sum(lda.n_m_z[1]-ALPHA), 10+len(vecs[1]))

        # n_z_w
        wfreq = count_word_freq(docs)
        self.assertAlmostEqual(np.sum(lda.n_z_w[:, 0]),
            wfreq[0] + K * BETA)
        self.assertAlmostEqual(np.sum(lda.n_z_w[:, 1]),
            wfreq[1] + K * BETA)

        # n_z
        self.assertAlmostEqual(lda.n_z[0], np.sum(lda.n_z_w[0, :]))
        self.assertAlmostEqual(lda.n_z[1], np.sum(lda.n_z_w[1, :]))

        # z_m_n, z_m_v
        self.assertAlmostEqual(list(lda.z_m_n[0]).count(0)
            + list(lda.z_m_v[0]).count(0), lda.n_m_z[0, 0]-ALPHA)
        self.assertAlmostEqual(list(lda.z_m_n[0]).count(1)
            + list(lda.z_m_v[0]).count(1), lda.n_m_z[0, 1]-ALPHA)
        for idx, vec in enumerate(vecs):
            self.assertEqual(len(vec), len(lda.z_m_v[idx]))

        # n_z_v
        for k in range(K):
            self.assertEqual(lda.n_z_v[k], np.sum([list(vec_topics).count(k)
                for vec_topics in lda.z_m_v]))

        # mu_z
        mu_0 = np.sum([np.sum([v for t, (l, v) in zip(ts, vs) 
            if t == 0], axis=0) for ts, vs in zip(lda.z_m_v, vecs)], axis=0)
        for c in range(L):
            self.assertAlmostEqual(lda.mu_z[0][c], mu_0[c])

        # sigma_z
        sigma_0 = np.identity(L)
        sigma_0 += np.sum([np.sum([np.outer(v, v) for t, (l, v) in zip(ts, vs)
            if t == 0], axis=0) for ts, vs in zip(lda.z_m_v, vecs)], axis=0)
        sigma_0 += -np.outer(mu_0, mu_0) / (KAPPA + lda.n_z_v[0])
        for c in range(L):
            self.assertAlmostEqual(sigma_0[c, c], lda.sigma_z[0][c, c])
            for c2 in range(L):
                self.assertAlmostEqual(sigma_0[c, c2], lda.sigma_z[0][c, c2])

if __name__ == '__main__':
    nose.main(argv=['nose', '-v'])
