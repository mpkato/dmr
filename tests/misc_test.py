# -*- coding:utf-8 -*-
import unittest
import numpy as np
from dmr.misc import gamma, gammaln
import scipy.special as special

class MiscTestCase(unittest.TestCase):
    def test_gamma(self):
        '''
        dmr.misc.gamma
        '''
        for x in range(1000):
            x += 1
            ideal = special.gamma(x/10.0)
            actual = gamma(x/10.0)
            self.assertAlmostEqual(np.abs(ideal - actual)/ ideal, 0)

    def test_gamma_speed(self):
        '''
        dmr.misc.gama is not 10 times as slow as the original one
        '''
        import time
        start = time.time()
        for x in range(1000):
            x += 1
            ideal = special.gamma(x/10.0)
        end = time.time()
        original_speed = end - start
        start = time.time()
        for x in range(1000):
            x += 1
            actual = gamma(x/10.0)
        end = time.time()
        new_speed = end - start

        self.assertLess(new_speed / original_speed, 10)

    def test_gammaln(self):
        '''
        dmr.misc.gammaln
        '''
        for x in range(1000):
            x += 1
            ideal = special.gammaln(x/10.0)
            actual = gammaln(x/10.0)
            print(x/10.0, ideal, actual, ideal - actual)
            if ideal != 0:
                self.assertAlmostEqual(np.abs(ideal - actual)/ ideal, 0)

    def test_gammalnexp(self):
        '''
        dmr.misc.gammaln
        '''
        for x in range(1000):
            x += 1
            ideal = special.gammaln(x/10.0)
            actual = gammaln(x/10.0)
            print(x/10.0, ideal, actual, ideal - actual)
            if ideal != 0:
                self.assertAlmostEqual(np.abs(ideal - actual)/ ideal, 0)

