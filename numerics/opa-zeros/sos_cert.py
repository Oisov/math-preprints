"""Numerical (not a proof). Joint SOS feasibility for the pointwise certificate

  Re(g_p(h) s) - c1 Re(s h^2) - sum_{m=2}^{p} Re(F_m(s) h^m) - omega(s) y^2  is SOS in (x,y)

for s = S_w(e^{i theta}) on a theta-sample, with F_m(s) = sum_{n=0}^{d} f_{m,n} conj(s)^n
(analytic in xi, so int F_m H^m dm = 0 when H(0)=0), real coefficients (symmetry, w real),
and closing condition omega >= 4 (1+delta) c1 (sigma-1)^2/sigma, c1 >= 0
(then int omega y^2 + c1 Re H(w)^2 > 0, see the depth remark of the paper).  p must be an even integer >= 4.
Usage: sos_cert.py p [d] [nth] [mmax]   -> bisection on w.
"""
import sys
import numpy as np
import sympy as sp
import cvxpy as cp

p = int(sys.argv[1]) if len(sys.argv) > 1 else 6
d = int(sys.argv[2]) if len(sys.argv) > 2 else 2
nth = int(sys.argv[3]) if len(sys.argv) > 3 else 120
mmax = int(sys.argv[4]) if len(sys.argv) > 4 else p
delta = 1e-3

x, y = sp.symbols("x y", real=True)
h = x + sp.I * y
hb = x - sp.I * y
E = 1 - h
g = sp.expand((1 - (E * (1 - hb)) ** ((p - 2) // 2) * E) * hb)
Reg, Img = sp.expand(sp.re(g)), sp.expand(sp.im(g))
deg = p
mons = [(i, j) for t in range(1, deg // 2 + 1) for i in range(t, -1, -1) for j in [t - i]]
nb = len(mons)
allmons = sorted({(a[0] + b[0], a[1] + b[1]) for a in mons for b in mons})
idx = {m: k for k, m in enumerate(allmons)}


def vec(expr):
    v = np.zeros(len(allmons))
    for (i, j), c in sp.Poly(sp.expand(expr), x, y).terms():
        if (i, j) not in idx:
            assert abs(float(c)) < 1e-14, (i, j, c)
            continue
        v[idx[(i, j)]] = float(c)
    return v


vRg, vIg = vec(Reg), vec(Img)
vRm = {m: vec(sp.re(sp.expand(h ** m))) for m in range(2, mmax + 1)}
vIm = {m: vec(sp.im(sp.expand(h ** m))) for m in range(2, mmax + 1)}
vy2 = vec(y ** 2)
# Gram map: coefficient of monomial = sum over pairs
A = np.zeros((len(allmons), nb, nb))
for a in range(nb):
    for b in range(nb):
        A[idx[(mons[a][0] + mons[b][0], mons[a][1] + mons[b][1])], a, b] = 1


def feasible(w, verbose=False, tight=False):
    th = np.pi * (np.arange(nth) + 0.5) / nth
    s = 1 / (1 - w * np.exp(-1j * th))
    sig = s.real
    c1 = cp.Variable(nonneg=True)
    f = {(m, n): cp.Variable() for m in range(2, mmax + 1) for n in range(d + 1)}
    om = cp.Variable(nth)
    t = cp.Variable()
    floor = 4 * (1 + delta) * c1 * (sig - 1) ** 2 / sig
    cons = [om == floor if tight else om >= floor, t <= 1]
    for k in range(nth):
        sk = s[k]
        # Re(a h^m) = Re(a) Re(h^m) - Im(a) Im(h^m)
        coef = sig[k] * vRg - sk.imag * vIg - c1 * (sk.real * vRm[2] - sk.imag * vIm[2]) - om[k] * vy2
        for (m, n), v in f.items():
            a = np.conj(sk) ** n
            coef = coef - v * (a.real * vRm[m] - a.imag * vIm[m])
        Q = cp.Variable((nb, nb), symmetric=True)
        cons += [Q >> t * np.eye(nb)]
        Gc = cp.hstack([cp.sum(cp.multiply(A[r], Q)) for r in range(len(allmons))])
        cons += [Gc == coef]
    prob = cp.Problem(cp.Maximize(t), cons)
    prob.solve(solver=cp.CLARABEL)
    if verbose:
        print(w, prob.status, t.value, c1.value, {k: round(float(v.value), 5) for k, v in f.items()})
    return prob.status == "optimal" and t.value is not None and t.value > 1e-7, (c1.value, f, om.value)


if __name__ == "__main__":
    lo, hi = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5, 0.999
    for _ in range(14):
        mid = (lo + hi) / 2
        ok, _ = feasible(mid, verbose=True)
        lo, hi = (mid, hi) if ok else (lo, mid)
    print(f"p={p} d={d} nth={nth} mmax={mmax}: largest feasible w ~ {lo:.5f}")
