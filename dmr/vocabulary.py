#!/usr/bin/env python
# -*- coding: utf-8 -*-

# This code is available under the MIT License.
# (c)2010-2011 Nakatani Shuyo / Cybozu Labs Inc.

class Vocabulary:
    """
    Manage vocabulary
    """

    def __init__(self):
        self.vocas = []    # id to word
        self.vocas_id = {} # word to id
        self.docfreq = []  # id to document frequency

    def read_corpus(self, corpus):
        result = []
        for doc in corpus:
            result.append(self.doc_to_ids(doc))
        return result

    def term_to_id(self, term):
        if term not in self.vocas_id:
            voca_id = len(self.vocas)
            self.vocas_id[term] = voca_id
            self.vocas.append(term)
            self.docfreq.append(0)
        else:
            voca_id = self.vocas_id[term]
        return voca_id

    def doc_to_ids(self, doc):
        result = []
        words = dict()
        for term in doc:
            tid = self.term_to_id(term)
            result.append(tid)
            if not tid in words:
                words[tid] = 1
                self.docfreq[tid] += 1
        return result

    def cut_low_freq(self, corpus, threshold=1):
        new_vocas = []
        new_docfreq = []
        self.vocas_id = dict()
        conv_map = dict()
        for tid, term in enumerate(self.vocas):
            freq = self.docfreq[tid]
            if freq > threshold:
                new_id = len(new_vocas)
                self.vocas_id[term] = new_id
                new_vocas.append(term)
                new_docfreq.append(freq)
                conv_map[tid] = new_id
        self.vocas = new_vocas
        self.docfreq = new_docfreq

        def conv(doc):
            new_doc = []
            for id in doc:
                if id in conv_map:
                    new_doc.append(conv_map[id])
            return new_doc

        return [conv(doc) for doc in corpus]

    def __getitem__(self, v):
        return self.vocas[v]

    def size(self):
        return len(self.vocas)
