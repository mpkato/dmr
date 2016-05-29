# -*- coding:utf-8 -*-
import unittest
import nose
import dmr
import os
import numpy as np
from collections import defaultdict

class DMRTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.docfilepath = os.path.join(
            os.path.dirname(__file__), '..', 'dat', 'DMR.doc.dat')
        self.vecfilepath = os.path.join(
            os.path.dirname(__file__), '..', 'dat', 'DMR.vec.dat')
        if not os.path.exists(self.docfilepath)\
            or not os.path.exists(self.vecfilepath):
            self._mkdat()
        self.K = 5
        self.sigma = 1.0
        self.beta = 0.01

    def _mkdat(self):
        M = 100
        N = 10
        K = 5
        L = 10
        V = 10
        CHAR_OFFSET = 97
        docs = []
        vecs = []
        for m in range(M):
            k = np.random.randint(0, K)
            # doc
            doc = []
            for n in range(N):
                v = np.random.randint(0, V)
                w = "%s%s" % (chr(CHAR_OFFSET+k), chr(CHAR_OFFSET+v))
                doc.append(w)
            docs.append(doc)
            # vec
            vec = np.zeros(L)
            vec[k] = 1.0
            vec[k+5] = 1.0
            vecs.append(vec)

        dirpath = os.path.dirname(self.docfilepath)
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        with open(self.docfilepath, "w") as f:
            for doc in docs:
                f.write(" ".join(doc) + "\n")
        with open(self.vecfilepath, "w") as f:
            for vec in vecs:
                f.write(" ".join(map(str, list(vec))) + "\n")

    def _read_vecfile(self, filepath):
        result = []
        with open(filepath, "r") as f:
            for line in f:
                ls = [float(l.strip()) for l in line.split(" ")]
                vec = np.array(ls, dtype=np.float32)
                result.append(vec)
        result = np.array(result, dtype=np.float32)
        return result

    def _init_dmr(self):
        corpus = dmr.Corpus.read(self.docfilepath)
        voca = dmr.Vocabulary()
        docs = voca.read_corpus(corpus)
        vecs = self._read_vecfile(self.vecfilepath)
        lda = dmr.DMR(self.K, self.sigma, self.beta, docs, vecs, voca.size())
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
        wfreq = self._count_word_freq(docs)
        self.assertAlmostEqual(np.sum(lda.n_z_w[:, 0]),
            wfreq[0] + self.K * self.beta)
        self.assertAlmostEqual(np.sum(lda.n_z_w[:, 1]),
            wfreq[1] + self.K * self.beta)

        # n_z
        self.assertAlmostEqual(lda.n_z[0], np.sum(lda.n_z_w[0, :]))
        self.assertAlmostEqual(lda.n_z[1], np.sum(lda.n_z_w[1, :]))

        # z_m_n
        self.assertAlmostEqual(list(lda.z_m_n[0]).count(0), lda.n_m_z[0, 0])
        self.assertAlmostEqual(list(lda.z_m_n[0]).count(1), lda.n_m_z[0, 1])

    def _count_word_freq(self, docs):
        result = defaultdict(int)
        for doc in docs:
            for w in doc:
                result[w] += 1
        return result

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


if __name__ == '__main__':
    nose.main(argv=['nose', '-v'])
