# -*- coding:utf-8 -*-
import unittest
import nose
import dmr
import os
import numpy as np
from collections import defaultdict
from tests.settings import (LDA_DOC_FILEPATH,
    K, ALPHA, BETA, mk_lda_dat)

class LDATestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)
        if not os.path.exists(LDA_DOC_FILEPATH):
            mk_lda_dat()

    def test_corpus_read(self):
        '''
        Corpus.read
        '''
        corpus = dmr.Corpus.read(LDA_DOC_FILEPATH)
        self.assertEqual(tuple(list(corpus)[0]),
            tuple(['ef', 'ea', 'ed', 'ed', 'eh', 'ej', 'ed', 'ef', 'ec', 'ee']))
        self.assertEqual(len(corpus), 100)

    def test_vocabulary_read_corpus(self):
        '''
        Vocabulary.read_corpus
        '''
        corpus = dmr.Corpus.read(LDA_DOC_FILEPATH)
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
        corpus = dmr.Corpus.read(LDA_DOC_FILEPATH)
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

    def _init_lda(self):
        corpus = dmr.Corpus.read(LDA_DOC_FILEPATH)
        voca = dmr.Vocabulary()
        docs = voca.read_corpus(corpus)
        lda = dmr.LDA(K, ALPHA, BETA, docs, voca.size())
        return voca, docs, lda

    def test_lda___init__(self):
        '''
        LDA.__init__
        '''
        voca, docs, lda = self._init_lda()

        # n_m_z
        self.assertAlmostEqual(np.sum(lda.n_m_z[0]), 10 + K * ALPHA)
        self.assertAlmostEqual(np.sum(lda.n_m_z[1]), 10 + K * ALPHA)

        # n_z_w
        wfreq = self._count_word_freq(docs)
        self.assertAlmostEqual(np.sum(lda.n_z_w[:, 0]),
            wfreq[0] + K * BETA)
        self.assertAlmostEqual(np.sum(lda.n_z_w[:, 1]),
            wfreq[1] + K * BETA)

        # n_z
        self.assertAlmostEqual(lda.n_z[0], np.sum(lda.n_z_w[0, :]))
        self.assertAlmostEqual(lda.n_z[1], np.sum(lda.n_z_w[1, :]))

        # z_m_n
        self.assertAlmostEqual(list(lda.z_m_n[0]).count(0) + ALPHA,
            lda.n_m_z[0, 0])
        self.assertAlmostEqual(list(lda.z_m_n[0]).count(1) + ALPHA,
            lda.n_m_z[0, 1])

    def _count_word_freq(self, docs):
        result = defaultdict(int)
        for doc in docs:
            for w in doc:
                result[w] += 1
        return result

    def test_lda_inference(self):
        '''
        LDA.inference
        '''
        voca, docs, lda = self._init_lda()

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
