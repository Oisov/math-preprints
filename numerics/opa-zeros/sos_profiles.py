"""Numerical (not a proof). Local obstruction for pointwise certificates.

For a boundary profile G in H^2(upper half-plane) put I(G) = int_R g_p(G(t)) dt,
g_p(h) = (1 - |1-h|^{p-2}(1-h)) conj(h).  For G = C/(1-it) this is pi |C|^2 W_p(C)
(paper, Lemma 14), so cos sup_C |arg W_p(C)| is the two-peak radius.  Any certificate
whose free terms integrate to zero on such profiles can certify at most
cos sup_G |arg I(G)|.  We compare sup over C with sup over G = sum_j c_j (1-it)^{-j}.
Usage: sos_profiles.py p [J]
"""
import sys
import numpy as np
from scipy.optimize import minimize

p = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
J = int(sys.argv[2]) if len(sys.argv) > 2 else 3
M = 4096
th = 2 * np.pi * (np.arange(M) + 0.5) / M - np.pi
t = np.tan(th / 2)
jac = (1 + t * t) / 2          # dt = jac dtheta
Hb = 1 / (1 - 1j * t)


def I(c):
    G = sum(c[j] * Hb ** (j + 1) for j in range(len(c)))
    E = 1 - G
    g = (1 - np.abs(E) ** (p - 2) * E) * np.conj(G)
    return np.sum(g * jac) * (2 * np.pi / M)


def negarg(x, n):
    c = x[:n] + 1j * x[n:]
    v = I(c)
    return -abs(np.angle(v))


rng = np.random.default_rng(0)
xb = None
for n in range(1, J + 1):
    best = 0
    for trial in range(20 if n == 1 else 8):
        if n == 1:
            x0 = rng.normal(size=2) * 1.5
        else:  # warm start from the (n-1)-term optimum
            x0 = np.concatenate([xb[:n - 1], [0], xb[n - 1:], [0]]) + rng.normal(size=2 * n) * 0.05
        r = minimize(negarg, x0, args=(n,), method="BFGS", options={"gtol": 1e-10})
        if -r.fun > best:
            best, xn = -r.fun, r.x
    xb = xn
    print(f"p={p} profiles with {n} term(s): sup|arg I| = {best:.8f}, radius cos = {np.cos(best):.6f}")
print("coefficients:", np.round(xb[:J] + 1j * xb[J:], 5))
print(f"Theorem 3 bound 2sqrt(p-1)/p = {2 * np.sqrt(p - 1) / p:.6f}")
