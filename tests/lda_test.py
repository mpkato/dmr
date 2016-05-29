# -*- coding:utf-8 -*-
import unittest
import nose
import dmr

class LDATestCase(unittest.TestCase):
    def test___init__(self):
        '''
        __init__ test
        '''
        corpus = dmr.Corpus.read("./doc.txt")
        voca = dmr.Vocabulary()
        docs = voca.read_corpus(corpus)


if __name__ == '__main__':
    nose.main(argv=['nose', '-v'])
