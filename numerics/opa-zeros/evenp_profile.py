"""Numerical (not a proof). Profile ceiling cos sup_G |arg B_p(G)|, G = sum_{k=1}^n c_k H^k.

With t = tan(b), H(t) = zeta = (1 + e^{2ib})/2 and dt = db/|zeta|^2, so
B_p(G) = (1/pi) int_{-pi/2}^{pi/2} g_p(Phi(zeta)) / |zeta|^2 db,  Phi(z) = sum c_k z^k
(trapezoid rule, spectrally accurate).  Many random restarts at each n, warm starts from n-1.
Usage: evenp_profile.py p nmax [restarts]
"""
import sys
import numpy as np
from scipy.optimize import minimize

p = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 10
R = int(sys.argv[3]) if len(sys.argv) > 3 else 10
M = 2048
b = np.pi * (np.arange(M) + 0.5) / M - np.pi / 2
z = (1 + np.exp(2j * b)) / 2
w8 = (np.pi / M) / np.abs(z) ** 2 / np.pi


def B(c):
    G = np.polyval(np.concatenate([c[::-1], [0]]), z)
    E = 1 - G
    return np.sum((1 - np.abs(E) ** (p - 2) * E) * np.conj(G) * w8)


def f(x, n):
    return -abs(np.angle(B(x[:n] + 1j * x[n:])))


rng = np.random.default_rng(1)
best_all, xb = 0, None
for n in range(1, nmax + 1):
    best = 0
    starts = [rng.normal(size=2 * n) * 1.2 for _ in range(R)]
    if xb is not None:
        m = n - 1
        base = np.concatenate([xb[:m], [0], xb[m:], [0]])
        starts += [base + rng.normal(size=2 * n) * 0.02 for _ in range(R)]
    for x0 in starts:
        r = minimize(f, x0, args=(n,), method="BFGS", options={"gtol": 1e-11, "maxiter": 5000})
        if -r.fun > best:
            best, xn = -r.fun, r.x
    xb = xn
    print(f"p={p} n={n}: sup|arg B| = {best:.9f}, cos = {np.cos(best):.7f}", flush=True)
print("c =", np.round(xb[:nmax] + 1j * xb[nmax:], 5))
