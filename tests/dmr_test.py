# -*- coding:utf-8 -*-
import unittest
import nose
import dmr
import os
import numpy as np
from tests.settings import (DMR_DOC_FILEPATH, DMR_VEC_FILEPATH,
    K, L, BETA, SIGMA, mk_dmr_dat, count_word_freq)

class DMRTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        if not os.path.exists(DMR_DOC_FILEPATH)\
            or not os.path.exists(DMR_VEC_FILEPATH):
            mk_dmr_dat()

    def _init_dmr(self):
        corpus = dmr.Corpus.read(DMR_DOC_FILEPATH)
        vcorpus = dmr.Corpus.read(DMR_VEC_FILEPATH, dtype=float)
        vecs = np.array([[v for v in vec] for vec in vcorpus], dtype=np.float32)
        voca = dmr.Vocabulary()
        docs = voca.read_corpus(corpus)
        lda = dmr.DMR(K, SIGMA, BETA, docs, vecs, voca.size())
        return voca, docs, vecs, lda

    def test_dmr___init__(self):
        '''
        DMR.__init__
        '''
        voca, docs, vecs, lda = self._init_dmr()

        # n_m_z
        self.assertAlmostEqual(np.sum(lda.n_m_z[0]), 10)
        self.assertAlmostEqual(np.sum(lda.n_m_z[1]), 10)

        # n_z_w
        wfreq = count_word_freq(docs)
        self.assertAlmostEqual(np.sum(lda.n_z_w[:, 0]),
            wfreq[0] + K * BETA)
        self.assertAlmostEqual(np.sum(lda.n_z_w[:, 1]),
            wfreq[1] + K * BETA)

        # n_z
        self.assertAlmostEqual(lda.n_z[0], np.sum(lda.n_z_w[0, :]))
        self.assertAlmostEqual(lda.n_z[1], np.sum(lda.n_z_w[1, :]))

        # z_m_n
        self.assertAlmostEqual(list(lda.z_m_n[0]).count(0), lda.n_m_z[0, 0])
        self.assertAlmostEqual(list(lda.z_m_n[0]).count(1), lda.n_m_z[0, 1])

    def test_dmr_inference(self):
        '''
        DMR.inference
        '''
        voca, docs, vecs, lda = self._init_dmr()

        n_m_z_0 = np.sum(lda.n_m_z[0])
        n_m_z_1 = np.sum(lda.n_m_z[1])
        n_z_w_0 = np.sum(lda.n_z_w[:, 0])
        n_z_w_1 = np.sum(lda.n_z_w[:, 1])

        lda.inference()

        self.assertAlmostEquals(np.sum(lda.n_m_z[0]), n_m_z_0)
        self.assertAlmostEquals(np.sum(lda.n_m_z[1]), n_m_z_1)
        self.assertAlmostEquals(np.sum(lda.n_z_w[:, 0]), n_z_w_0)
        self.assertAlmostEquals(np.sum(lda.n_z_w[:, 1]), n_z_w_1)

    def test_dmr__ll(self):
        '''
        DMR.inference
        '''
        voca, docs, vecs, lda = self._init_dmr()
        
        Lambda = np.random.multivariate_normal(np.zeros(L), 
            (SIGMA ** 2) * np.identity(L), size=K)
        ideal = lda._ll(Lambda)
        actual = lda._new_ll(Lambda)
        self.assertAlmostEqual(actual, ideal)

if __name__ == '__main__':
    nose.main(argv=['nose', '-v'])
