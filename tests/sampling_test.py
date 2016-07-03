# -*- coding:utf-8 -*-
import unittest
import pyximport; pyximport.install()
import numpy as np
pyximport.install(setup_args={"include_dirs":np.get_include()})
from dmr.sampling1 import *
from collections import defaultdict
import time

class SamplingTestCase(unittest.TestCase):
    def setUp(self):
        np.random.seed(0)

    def test_sampling(self):
        '''
        dmr.sampling.gen_alias_table, dmr.sampling.alias_sampling
        '''
        x = np.random.rand(500)
        x /= np.sum(x)
        def numpy_sampling(p):
            return np.random.multinomial(1, p).argmax()
        table = gen_alias_table(x)
        print(table)
        def new_sampling(p):
            return alias_sampling(table)

        NUM = 100000
        for func in [numpy_sampling, new_sampling]:
            start = time.time()
            res = self._repeat_sampling(numpy_sampling, x, NUM)
            end = time.time()
            ellapsed = end - start
            print(res)
            print(func, ellapsed)
        raise


    def _repeat_sampling(self, func, arg, num):
        result = defaultdict(float)
        for i in range(num):
            r = func(arg)
            result[r] += 1
        return result
