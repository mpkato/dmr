import os
# lda

# dmr
DMR_DOC_FILEPATH = os.path.join(
    os.path.dirname(__file__), '..', 'dat', 'DMR.doc.dat')
DMR_VEC_FILEPATH = os.path.join(
    os.path.dirname(__file__), '..', 'dat', 'DMR.vec.dat')
L = 10
SIGMA = 1.0

# shared
CHAR_OFFSET = 97
K = 5
M = 100
N = 10
V = 10
BETA = 0.01

def mk_dmr_dat():
    docs = []
    vecs = []
    for m in range(M):
        k = np.random.randint(0, K)
        # doc
        doc = []
        for n in range(N):
            v = np.random.randint(0, V)
            w = "%s%s" % (chr(CHAR_OFFSET+k), chr(CHAR_OFFSET+v))
            doc.append(w)
        docs.append(doc)
        # vec
        vec = np.zeros(L)
        vec[k] = 1.0
        vec[k+5] = 1.0
        vecs.append(vec)

    dirpath = os.path.dirname(DMR_DOC_FILEPATH)
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    with open(DMR_DOC_FILEPATH, "w") as f:
        for doc in docs:
            f.write(" ".join(doc) + "\n")
    with open(DMR_VEC_FILEPATH, "w") as f:
        for vec in vecs:
            f.write(" ".join(map(str, list(vec))) + "\n")

