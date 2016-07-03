import numpy as np

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

def alias_sampling(table):
    l = len(table[0])
    r = np.random.rand()
    i = int(l * r)
    p, h = table[0][i], table[1][i]
    if l * r - i > p * l:
        return h
    else:
        return i
