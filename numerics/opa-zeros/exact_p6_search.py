"""Numerical search (NOT a proof) for the p = 6 certificate of Proposition 2 of the paper.

Find c1 >= 0, real f_{m,n} (2 <= m <= 6, 0 <= n <= d) and SOS Gram matrices Q0, Q1 with

  G := sigma*[Re(g(h)s) - c1 Re(s h^2) - sum f_{m,n} Re(conj(s)^n h^m)] - kappa (sigma-1)^2 y^2
     = z0' Q0 z0 + (2 sigma - 1 - (1-r^2)(sigma^2+tau^2)) z1' Q1 z1,      kappa = 4(1+delta) c1,

as polynomials in (x, y, sigma, tau), h = x+iy, s = sigma+i tau.  The second factor is >= 0
exactly on the closed disc {s : |s-1| <= r|s|} = union of the circles S_w(T), |w| <= r.
Both Gram matrices are block-diagonal by the symmetry (y, tau) -> (-y, -tau).

Usage: exact_p6_search.py r d D0 D1 [p]     (p even, default 6; g_p and m <= p, h-degree <= p/2)
  r: radius (rational string, e.g. 17/20), d: degree of the s-bar multipliers,
  D0/D1: s-degree of the monomial bases of Q0/Q1.  Prints the max-min-eigenvalue margin.
  exact_p6_round.py turns a solution into exact certificate data.
"""
import sys
from fractions import Fraction
import numpy as np
import sympy as sp
import cvxpy as cp

x, y, S, T = sp.symbols("x y sigma tau", real=True)
VARS = (x, y, S, T)
h, hb = x + sp.I * y, x - sp.I * y
s, sb = S + sp.I * T, S - sp.I * T
P = 6  # the exponent p (even); set by main() from the optional fifth argument


def g_p(p):
    """g_p(h) = (1 - |1-h|^{p-2}(1-h)) conj(h)."""
    E, Eb = 1 - h, 1 - hb
    return sp.expand((1 - (E * Eb) ** ((p - 2) // 2) * E) * hb)


def re(e):
    return sp.expand((sp.expand(e) + sp.expand(sp.conjugate(e))) / 2)


def parts(d):
    """G = G0 + c1*Gc + sum f[m,n]*G[m,n]  (kappa = 4(1+delta)c1 folded into Gc)."""
    G0 = sp.expand(S * re(g_p(P) * s))
    Gc = sp.expand(-S * re(s * h**2))
    Gk = sp.expand(-(S - 1) ** 2 * y**2)
    Gf = {(m, n): sp.expand(-S * re(sb**n * h**m)) for m in range(2, P + 1) for n in range(d + 1)}
    return G0, Gc, Gk, Gf


def basis(D):
    hm = [(a, t - a) for t in range(1, P // 2 + 1) for a in range(t, -1, -1)]
    sm = [(c, k - c) for k in range(D + 1) for c in range(k, -1, -1)]
    B = [(a, b, c, e) for (a, b) in hm for (c, e) in sm]
    return [[m for m in B if (m[1] + m[3]) % 2 == par] for par in (0, 1)]


def mono(m):
    return x ** m[0] * y ** m[1] * S ** m[2] * T ** m[3]


def coeffs(e):
    return {k: v for k, v in sp.Poly(sp.expand(e), *VARS).terms()}


def solve(r, d, D0, D1, fixed=None, delta=Fraction(1, 1000)):
    """Max-margin SDP.  fixed = (c1, f) with rationals fixes the multipliers.
    Returns (margin, c1, f, [Q blocks: B0 even, B0 odd, B1 even, B1 odd], bases)."""
    import scipy.sparse as ssp
    rr = sp.Rational(r.numerator, r.denominator)
    mq = sp.expand(2 * S - 1 - (1 - rr**2) * (S**2 + T**2))
    G0, Gc, Gk, Gf = parts(d)
    kap = 4 * (1 + float(delta))
    cG0, cGc, cGk = coeffs(G0), coeffs(Gc), coeffs(Gk)
    cGf = {k: coeffs(v) for k, v in Gf.items()}
    B0, B1 = basis(D0), basis(D1)
    cq = coeffs(mq)
    allm = set(cG0) | set(cGc) | set(cGk)
    for v in cGf.values():
        allm |= set(v)
    for blocks, mult in ((B0, {(0, 0, 0, 0): 1}), (B1, cq)):
        for blk in blocks:
            for a in blk:
                for b in blk:
                    for qm in mult:
                        allm.add(tuple(i + j + k for i, j, k in zip(a, b, qm)))
    allm = sorted(allm)
    idx = {m: i for i, m in enumerate(allm)}
    M = len(allm)

    def vec(c):
        v = np.zeros(M)
        for k, val in c.items():
            v[idx[k]] = float(val)
        return v

    if fixed is None:
        c1 = cp.Variable(nonneg=True)
        f = {k: cp.Variable() for k in Gf}
    else:
        c1 = float(fixed[0])
        f = {k: float(fixed[1][k]) for k in Gf}
    rhs = vec(cG0) + c1 * (vec(cGc) + kap * vec(cGk))
    for k in Gf:
        rhs = rhs + f[k] * vec(cGf[k])
    t = cp.Variable()
    cons, lhs, Qs = [], 0, []
    for blocks, mult in ((B0, {(0, 0, 0, 0): 1}), (B1, cq)):
        for blk in blocks:
            n = len(blk)
            Q = cp.Variable((n, n), symmetric=True)
            Qs.append(Q)
            cons.append(Q >> t * np.eye(n))
            rows, cols, vals = [], [], []
            for i, a in enumerate(blk):
                for j, b in enumerate(blk):
                    for qm, qc in mult.items():
                        rows.append(idx[tuple(p + q + w for p, q, w in zip(a, b, qm))])
                        cols.append(i * n + j)
                        vals.append(float(qc))
            A = ssp.csr_matrix((vals, (rows, cols)), shape=(M, n * n))
            lhs = lhs + A @ cp.vec(Q, order="C")
    cons += [lhs == rhs, t <= 1]
    if fixed is None:
        cons.append(c1 <= 10)
        cons += [cp.abs(v) <= 10 for v in f.values()]
    prob = cp.Problem(cp.Maximize(t), cons)
    prob.solve(solver=cp.CLARABEL)
    c1v = c1.value if fixed is None else c1
    fv = {k: (v.value if fixed is None else v) for k, v in f.items()}
    return t.value, c1v, fv, [Q.value for Q in Qs], (B0, B1)


def main():
    global P
    r = Fraction(sys.argv[1])
    d, D0, D1 = map(int, sys.argv[2:5])
    P = int(sys.argv[5]) if len(sys.argv) > 5 else 6
    t, c1, f, Qs, _ = solve(r, d, D0, D1)
    print(f"r={r} d={d} D0={D0} D1={D1}: margin t={t} c1={c1}")


if __name__ == "__main__":
    main()
