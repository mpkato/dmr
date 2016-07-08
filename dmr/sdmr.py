#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import scipy.special as special
import scipy.optimize as optimize
import scipy.misc as misc
from .dmr import DMR

class SDMR(DMR):
    '''
    Simple Topic Model with Dirichlet Multinomial Regression
    '''
    def get_alpha_n_m_z(self, idx=None):
        if idx is None:
            return self.alpha
        else:
            return self.alpha[idx]
