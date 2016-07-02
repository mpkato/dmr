import numpy as np
P0 = 0.99999999999980993
P = np.array([ 676.5203681218851,   -1259.1392167224028,  771.32342877765313,
    -176.61502916214059,     12.507343278686905, -0.13857109526572012,
    9.9843695780195716e-6, 1.5056327351493116e-7])
E = np.exp(1)
I = np.array(np.arange(P.shape[0]))

def gamma(z):
    if z < 0.5:
        result = np.pi / (np.sin(np.pi*z) * gamma(1-z))
    else:
        z -= 1
        x = P0 + np.dot(P, 1.0 / (z + I + 1))
        t = z + P.shape[0] - 0.5
        result = np.sqrt(2 * np.pi) * t**(z + 0.5) * np.exp(-t) * x
    return result

def gammaln(z):
    if z < 0.5:
        result = np.log(np.pi) - np.log(np.sin(np.pi*z)) - gammaln(1-z)
    else:
        z -= 1
        x = P0 + np.dot(P, 1.0 / (z + I + 1))
        t = z + P.shape[0] - 0.5
        result = np.log(2 * np.pi) / 2 + (z + 0.5) * np.log(t) - t + np.log(x)
    return result

def gammaln(z):
    if z < 0.5:
        result = np.log(np.pi) - np.log(np.sin(np.pi*z)) - gammaln(1-z)
    else:
        z -= 1
        x = P0 + np.dot(P, 1.0 / (z + I + 1))
        t = z + P.shape[0] - 0.5
        result = np.log(2 * np.pi) / 2 + (z + 0.5) * np.log(t) - t + np.log(x)
    return result
