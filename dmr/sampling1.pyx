import numpy as np
import sys
cimport numpy as np
cimport cython
@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire

def gen_alias_table(p):
    l = p.shape[0]
    invl = 1.0 / l
    L, H, = [], []
    A, B = np.zeros(l, dtype=np.float32), np.zeros(l, dtype=np.int32)
    for i in range(l):
        if p[i] <= invl:
            L.append((i, p[i]))
        else:
            H.append((i, p[i]))
    while len(L) != 0:
        if len(H) == 0:
            for i, pi in L:
                A[i] = invl
                B[i] = -1
            break
        i, pi = L.pop()
        h, ph = H.pop()
        A[i] = pi
        B[i] = h
        p = ph - (invl - pi)
        if p > invl:
            H.append((h, p))
        elif p < invl:
            L.append((h, p))
        else:
            A[i] = p
            B[i] = h
    return A, B

cdef alias_sampling(np.ndarray[np.float_t, ndim=1] A, np.ndarray[np.int_t, ndim=1] B):
    cdef int l, i, h
    cdef float r, p
    l = len(A)
    r = np.random.rand()
    i = int(l * r)
    p, h = A[i], B[i]
    if l * r - i > p * l:
        return h
    else:
        return i
