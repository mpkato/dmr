# -*- coding:utf-8 -*-
import unittest
import nose
import dmr
import os
import numpy as np
from collections import defaultdict

class LDATestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        self.datfilepath = os.path.join(
            os.path.dirname(__file__), '..', 'dat', 'LDA.dat')
        if not os.path.exists(self.datfilepath):
            self._mkdat()
        self.K = 5
        self.alpha = 0.1
        self.beta = 0.01


    def _mkdat(self):
        M = 100
        N = 10
        K = 5
        V = 10
        CHAR_OFFSET = 97
        docs = []
        for m in range(M):
            k = np.random.randint(0, K)
            doc = []
            for n in range(N):
                v = np.random.randint(0, V)
                w = "%s%s" % (chr(CHAR_OFFSET+k), chr(CHAR_OFFSET+v))
                doc.append(w)
            docs.append(doc)

        dirpath = os.path.dirname(self.datfilepath)
        if not os.path.exists(dirpath):
            os.mkdir(dirpath)
        with open(self.datfilepath, "w") as f:
            for doc in docs:
                f.write(" ".join(doc) + "\n")

    def test_corpus_read(self):
        '''
        Corpus.read
        '''
        corpus = dmr.Corpus.read(self.datfilepath)
        self.assertEqual(tuple(list(corpus)[0]),
            tuple(['ef', 'ea', 'ed', 'ed', 'eh', 'ej', 'ed', 'ef', 'ec', 'ee']))
        self.assertEqual(len(corpus), 100)

    def test_vocabulary_read_corpus(self):
        '''
        Vocabulary.read_corpus
        '''
        corpus = dmr.Corpus.read(self.datfilepath)
        voca = dmr.Vocabulary()
        docs = voca.read_corpus(corpus)
        self.assertEqual(docs[0][2], docs[0][3])
        self.assertEqual(docs[0][0], docs[0][7])
        self.assertEqual(voca[docs[0][2]], 'ed')
        self.assertEqual(voca[docs[0][0]], 'ef')

    def test_vocabulary_cut_low_freq(self):
        '''
        Vocabulary.cut_low_freq
        '''
        corpus = dmr.Corpus.read(self.datfilepath)
        voca = dmr.Vocabulary()
        docs = voca.read_corpus(corpus)
        freq = self._count_doc_freq(docs)
        cut_docs = voca.cut_low_freq(docs, 10)
        self.assertEqual(
            len([w for w in docs[0] if freq[w] > 10]),
            len(cut_docs[0])
            )
        self.assertEqual(
            len([w for w in docs[1] if freq[w] > 10]),
            len(cut_docs[1])
            )

    def _count_doc_freq(self, docs):
        result = defaultdict(int)
        for doc in docs:
            for w in set(doc):
                result[w] += 1
        return result

    def test___init__(self):
        '''
        __init__ test
        '''
        corpus = dmr.Corpus.read(self.datfilepath)
        voca = dmr.Vocabulary()
        docs = voca.read_corpus(corpus)
        lda = dmr.LDA(self.K, self.alpha, self.beta, docs, voca.size())

        # n_m_z
        self.assertAlmostEqual(np.sum(lda.n_m_z[0]), 10 + self.K * self.alpha)
        self.assertAlmostEqual(np.sum(lda.n_m_z[1]), 10 + self.K * self.alpha)

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
        self.assertAlmostEqual(list(lda.z_m_n[0]).count(0) + self.alpha,
            lda.n_m_z[0, 0])
        self.assertAlmostEqual(list(lda.z_m_n[0]).count(1) + self.alpha,
            lda.n_m_z[0, 1])

    def _count_word_freq(self, docs):
        result = defaultdict(int)
        for doc in docs:
            for w in doc:
                result[w] += 1
        return result



if __name__ == '__main__':
    nose.main(argv=['nose', '-v'])
