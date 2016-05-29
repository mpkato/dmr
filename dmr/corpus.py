#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re

class Corpus:
    """
    Reads a file (where each line represents a document and words are separated by white space).
    """

    def __init__(self, docs):
        self.docs = docs

    @classmethod
    def read(cls, filename):
        docs = []
        with open(filename, 'r') as f:
            for line in f:
                doc = [w.strip() for w in line.split(' ')]
                if len(doc) > 0:
                    docs.append(doc)
        return Corpus(docs)

    def __iter__(self):
        for doc in self.docs:
            yield doc

    def __len__(self):
        return len(self.docs)
