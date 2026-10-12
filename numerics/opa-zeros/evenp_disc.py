"""Numerical (not a proof). Disc ceiling: sup |arg int_T g_p(Phi) dm| over Phi(z) = sum_{k=1}^n a_k z^k.

A profile G = Phi(F), F = b^N chi (b inner, chi outer with |chi| ~ 1 on [-L, L]) has value law
~ 2L Phi_*(m) as N, L -> oo, so cos of this sup is also an upper bound for rho_1(p)
(same proof as the polynomial-profile bound, by the generalised two-peak lemma).
Usage: evenp_disc.py p nmax [restarts]
"""
import sys
import numpy as np
from scipy.optimize import minimize

p = float(sys.argv[1]) if len(sys.argv) > 1 else 6.0
nmax = int(sys.argv[2]) if len(sys.argv) > 2 else 10
R = int(sys.argv[3]) if len(sys.argv) > 3 else 10
M = 1024
z = np.exp(2j * np.pi * np.arange(M) / M)


def B(c):
    G = np.polyval(np.concatenate([c[::-1], [0]]), z)
    E = 1 - G
    return np.mean((1 - np.abs(E) ** (p - 2) * E) * np.conj(G))


def f(x, n):
    return -abs(np.angle(B(x[:n] + 1j * x[n:])))


rng = np.random.default_rng(2)
xb = None
for n in range(1, nmax + 1):
    best = 0
    starts = [rng.normal(size=2 * n) * 0.8 for _ in range(R)]
    if xb is not None:
        m = n - 1
        base = np.concatenate([xb[:m], [0], xb[m:], [0]])
        starts += [base + rng.normal(size=2 * n) * 0.02 for _ in range(R)]
    for x0 in starts:
        r = minimize(f, x0, args=(n,), method="BFGS", options={"gtol": 1e-11, "maxiter": 5000})
        if -r.fun > best:
            best, xn = -r.fun, r.x
    xb = xn
    print(f"p={p} n={n}: sup|arg| = {best:.9f}, cos = {np.cos(best):.7f}", flush=True)
print("a =", np.round(xb[:nmax] + 1j * xb[nmax:], 5))
