"""EXACT verification of the zero-radius certificates of Proposition 2 of the paper (p = 6, and p = 8 by the same
scheme; p is read from the data module, default 6).  Standard library only: all arithmetic is in
fractions.Fraction; no floating point is used in any check.  The formulas below are written for
p = 6; for general even p, g = (1-|1-h|^{p-2}(1-h)) conj(h), m <= p and h-degree <= p/2 in z0, z1.

Claim checked.  With h = x+iy, s = sigma+i tau, u = |h|^2, r, delta, c1, f_{m,n} from the data
module, kappa = 4(1+delta)c1 and
  g(h) = u^3+6u^2+3u - u(3h+6hb) + u(h^2+3hb^2) - u^2(2h+3hb) + 2hb^2 - hb^3   (= (1-|1-h|^4(1-h)) hb),
  G    = sigma*[Re(g s) - c1 Re(s h^2) - sum_{m,n} f_{m,n} Re(conj(s)^n h^m)] - kappa (sigma-1)^2 y^2,
  mq   = 2 sigma - 1 - (1-r^2)(sigma^2+tau^2)      (>= 0 iff |s-1| <= r|s|),
there are positive semidefinite rational matrices Q0, Q1 (block diagonal) with the polynomial
identity in Q[x, y, sigma, tau]
  G = z0' Q0 z0 + mq * z1' Q1 z1 .
Hence G >= 0 whenever mq >= 0, in particular for s on every circle S_w(T) with w real, |w| <= r.

Q1 and c1, f are read from the data module; Q0 is the stored rounded Gram matrix after an exact
orthogonal projection onto the affine space {Q : z0' Q z0 = G - mq z1' Q1 z1}, computed here.

Usage: python3 exact_p6_verify.py [data_module]     (default exact_p6_data_0p8607)
"""
import sys
import importlib
from fractions import Fraction as F

# ---------- polynomials in (x, y, sigma, tau): dict exponent-tuple -> Fraction ----------


def padd(*ps):
    out = {}
    for p in ps:
        for k, v in p.items():
            out[k] = out.get(k, 0) + v
    return {k: v for k, v in out.items() if v != 0}


def pscale(p, c):
    return {k: v * c for k, v in p.items() if v * c != 0}


def pmul(p, q):
    out = {}
    for a, u in p.items():
        for b, v in q.items():
            k = (a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3])
            out[k] = out.get(k, 0) + u * v
    return {k: v for k, v in out.items() if v != 0}


def mono(a, b, c, e):
    return {(a, b, c, e): F(1)}


ONE = mono(0, 0, 0, 0)
X, Y, SG, TA = mono(1, 0, 0, 0), mono(0, 1, 0, 0), mono(0, 0, 1, 0), mono(0, 0, 0, 1)

# complex polynomials: pairs (real part, imaginary part)


def cadd(*zs):
    return (padd(*[z[0] for z in zs]), padd(*[z[1] for z in zs]))


def cmul(z, w):
    return (padd(pmul(z[0], w[0]), pscale(pmul(z[1], w[1]), -1)), padd(pmul(z[0], w[1]), pmul(z[1], w[0])))


def cscale(z, c):
    return (pscale(z[0], c), pscale(z[1], c))


def conj(z):
    return (z[0], pscale(z[1], -1))


def cpow(z, n):
    out = (ONE, {})
    for _ in range(n):
        out = cmul(out, z)
    return out


def creal(p):
    return (p, {})


H = (X, Y)
HB = conj(H)
Sc = (SG, TA)
SB = conj(Sc)
U = creal(padd(pmul(X, X), pmul(Y, Y)))


def g_closed():
    u2, u3 = cmul(U, U), cpow(U, 3)
    return cadd(u3, cscale(u2, 6), cscale(U, 3),
                cscale(cmul(U, cadd(cscale(H, 3), cscale(HB, 6))), -1),
                cmul(U, cadd(cmul(H, H), cscale(cmul(HB, HB), 3))),
                cscale(cmul(u2, cadd(cscale(H, 2), cscale(HB, 3))), -1),
                cscale(cmul(HB, HB), 2), cscale(cpow(HB, 3), -1))


def g_definition(p=6):
    E = cadd((ONE, {}), cscale(H, -1))                     # 1 - h
    absE = cpow(cmul(E, conj(E)), (p - 2) // 2)            # |1-h|^{p-2}
    return cmul(cadd((ONE, {}), cscale(cmul(absE, E), -1)), HB)


# ---------- exact PSD test (symmetric Gaussian elimination / LDL') ----------


def psd(A):
    """True iff the rational symmetric matrix A is positive semidefinite; also returns #positive pivots."""
    A = [row[:] for row in A]
    n = len(A)
    pos = 0
    for k in range(n):
        p = A[k][k]
        if p < 0:
            return False, pos
        if p == 0:
            if any(A[k][j] != 0 for j in range(k + 1, n)):
                return False, pos
            continue
        pos += 1
        for i in range(k + 1, n):
            if A[i][k] != 0:
                l = A[i][k] / p
                for j in range(k + 1, n):
                    A[i][j] -= l * A[k][j]
    return True, pos


def basis(D, p=6):
    hm = [(a, t - a) for t in range(1, p // 2 + 1) for a in range(t, -1, -1)]
    sm = [(c, k - c) for k in range(D + 1) for c in range(k, -1, -1)]
    B = [(a, b, c, e) for (a, b) in hm for (c, e) in sm]
    return [[m for m in B if (m[1] + m[3]) % 2 == par] for par in (0, 1)]


def gram_poly(blocks, Qs):
    out = {}
    for blk, Q in zip(blocks, Qs):
        for i, a in enumerate(blk):
            for j, b in enumerate(blk):
                if Q[i][j] != 0:
                    k = tuple(p + q for p, q in zip(a, b))
                    out[k] = out.get(k, 0) + Q[i][j]
    return {k: v for k, v in out.items() if v != 0}


def main():
    name = sys.argv[1] if len(sys.argv) > 1 else "exact_p6_data_0p8607"
    D = importlib.import_module(name).DATA
    p = D.get("p", 6)
    r, delta, c1 = F(D["r"]), F(D["delta"]), F(D["c1"])
    d, D0, D1 = D["d"], D["D0"], D["D1"]
    f = {tuple(map(int, k.split(","))): F(v) for k, v in D["f"].items()}
    den = D["den"]
    Qr = [[[F(int(v), den) for v in row.split()] for row in Q] for Q in D["Q"]]
    kappa = 4 * (1 + delta) * c1
    print(f"data {name}: p = {p}, r = {r} = {float(r):.6f}, d = {d}, D0 = {D0}, D1 = {D1}")
    print(f"  c1 = {c1},  delta = {delta},  kappa = 4(1+delta)c1 = {kappa}")

    ok = True

    def check(cond, msg):
        nonlocal ok
        ok &= bool(cond)
        print(("  [ok]   " if cond else "  [FAIL] ") + msg)

    check(p % 2 == 0 and p >= 4 and all(2 <= m <= p and n >= 0 for m, n in f), "p even, multipliers Re(conj(s)^n h^m) with 2 <= m <= p")
    if p == 6:
        check(g_closed() == g_definition(), "g(h) closed form = (1-|1-h|^4(1-h)) conj(h)  (exact polynomial identity)")
    check(c1 > 0 and delta > 0, "c1 > 0 and kappa > 4 c1")

    # G
    g = g_definition(p)                                    # (1-|1-h|^{p-2}(1-h)) conj(h)
    reGs = cmul(g, Sc)[0]
    inner = padd(reGs, pscale(cmul(Sc, cmul(H, H))[0], -c1))
    for (m, n), v in f.items():
        inner = padd(inner, pscale(cmul(cpow(SB, n), cpow(H, m))[0], -v))
    sm1 = padd(SG, pscale(ONE, -1))
    G = padd(pmul(SG, inner), pscale(pmul(pmul(sm1, sm1), pmul(Y, Y)), -kappa))
    mq = padd(pscale(SG, 2), pscale(ONE, -1), pscale(padd(pmul(SG, SG), pmul(TA, TA)), -(1 - r * r)))
    check(all(m[0] + m[1] >= 2 for m in G), "G vanishes to second order at h = 0 (no constant/linear terms in h)")

    B0, B1 = basis(D0, p), basis(D1, p)
    Q0, Q1 = Qr[:2], Qr[2:]
    for k, Q in enumerate(Q1):
        good, rk = psd(Q)
        check(good and all(Q[i][j] == Q[j][i] for i in range(len(Q)) for j in range(len(Q))),
              f"Q1 block {k} ({len(Q)}x{len(Q)}) symmetric PSD (exact LDL, {rk} positive pivots)")
    P = padd(G, pscale(pmul(mq, gram_poly(B1, Q1)), -1))

    # exact projection of Q0 onto {z0' Q z0 = P}: spread each monomial residual evenly
    cells = {}
    for b, blk in enumerate(B0):
        for i, a in enumerate(blk):
            for j, c in enumerate(blk):
                cells.setdefault(tuple(p + q for p, q in zip(a, c)), []).append((b, i, j))
    missing = [k for k in P if k not in cells]
    check(not missing, f"every monomial of G - mq*z1'Q1z1 is representable by z0 ({len(P)} monomials)")
    cur = gram_poly(B0, Q0)
    for k, cl in cells.items():
        res = (P.get(k, 0) - cur.get(k, 0)) / len(cl)
        if res:
            for b, i, j in cl:
                Q0[b][i][j] += res
    check(padd(gram_poly(B0, Q0), pscale(pmul(mq, gram_poly(B1, Q1)), 1)) == G,
          "EXACT IDENTITY  G = z0'Q0z0 + mq * z1'Q1z1  in Q[x,y,sigma,tau]")
    for k, Q in enumerate(Q0):
        good, rk = psd(Q)
        check(good and rk == len(Q) and all(Q[i][j] == Q[j][i] for i in range(len(Q)) for j in range(len(Q))),
              f"Q0 block {k} ({len(Q)}x{len(Q)}) symmetric positive DEFINITE (exact LDL, {rk} positive pivots)")
    # the disc {mq >= 0} contains every circle S_w(T), |w| <= r: |s|^2 r^2 - |s-1|^2 = mq
    s2 = padd(pmul(SG, SG), pmul(TA, TA))
    check(padd(pscale(s2, r * r), pscale(padd(s2, pscale(SG, -2), ONE), -1)) == mq,
          "mq = r^2|s|^2 - |s-1|^2, so mq >= 0 on S_w(T) for real |w| <= r (|s-1| = |w||s| there)")
    print("RESULT:", "ALL CHECKS PASSED -- rho_1(%d) >= %s" % (p, r) if ok else "VERIFICATION FAILED")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
