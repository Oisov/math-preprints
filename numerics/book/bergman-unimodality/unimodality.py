"""Certificates for Theorem 6.19: Lemma 6.20 (certified values) and Lemma 6.22
(pointwise log-convexity of the dilated kernel), plus the slope inequalities in
the proof of Lemma 6.21.

Notation of Section 6.6: rho = (alpha-1)/2, Z(q) = zeta(1+q)^alpha - 1,
k_alpha(x,y) = Z(x+y) (xy)^rho, Lambda_alpha = Gamma(alpha/2)^2/Gamma(alpha),
c(alpha) = (alpha-1)^(2/3), Phi_q(alpha) = alpha log c + log Z(c q).

Part 1 (Lemma 6.20).
* N(alpha) <= U by Schur's test with w(y) = y^(-sigma) e^(-tau y):
  S(x) = x^(sigma+rho) e^(tau x) J(x),  J(x) = int_0^oo Z(x+y) g(y) dy,
  g(y) = y^beta e^(-tau y), beta = rho - sigma > 0.
  Z is completely monotone (a Dirichlet series with positive coefficients), so
  Z and J are convex and decreasing.
  - x <= 1e-8 and x >= 80: the analytic bounds of the book.
  - 1e-8 <= x <= 80, geometric x-grid of ratio 1.01.  J(x_i) is bounded above
    by: the head int_0^(1e-13) <= Z(x) (1e-13)^(1+beta)/(1+beta); the tail
    int_40^oo <= 2 alpha 2^-x int_40^oo y^2 e^(-(log 2 + tau) y) dy, from
    Z(q) <= 2 alpha 2^-q for q >= 2 and y^beta <= y^2 (as in the book);
    and on [1e-13, 40] a y-grid (geometric, ratio 1.01, up to y = 2, then step
    1/50), on whose cells Z(x+.) lies below its chord, so the cell integral is
    at most Z(x+y_k) G-_k + Z(x+y_{k+1}) G+_k with
    G-_k = int g(y)(y_{k+1}-y)/h_k, G+_k = int g(y)(y-y_k)/h_k, computed from
    incomplete gamma functions in Arb.  Values Z(x+y_k) are in turn bounded by
    chords of a table of Arb enclosures of Z on a geometric q-grid of ratio
    1.005 (again convexity).  The table lookup and the dot products are done in
    binary64: every rounded sum x + y_k is moved one binary64 step down before
    the decreasing Z is bounded; the chord offset Q_(j+1) - q is exact
    (Sterbenz); with u = 2^-53 a positive sum of n terms, each the result of at
    most three roundings, is at most the computed value divided by
    (1 - (n+3)u).  All intermediates are checked positive and normal.
  - On a cell [x1, x2] = [x1, x1 + h]: log(x^(sigma+rho) e^(tau x)) is concave
    (sigma + rho > 0), so lies below its tangent at x1; J lies below the chord of
    the upper bounds J1, J2, and 1 + t <= e^t.  Hence
        S(x) <= x1^(sigma+rho) e^(tau x1) J1 exp(max(d, 0) h),
        d = (sigma+rho)/x1 + tau + (J2/J1 - 1)/h.
* N(3) >= Rayleigh quotient of f(y) = y^(-s) e^(-t y):
  pref * int_0^oo Z(q) q^(alpha-2s) e^(-tq) dq; the integrand is positive, so
  dropping [0, 1e-6] and [60, oo) leaves a lower bound, and the integral over
  [1e-6, 60] is enclosed by rigorous Gauss-Legendre integration in Arb.
* Lambda_(8/5) in Arb.

Part 2 (Lemma 6.22, the range 1e-40 <= q <= 8, alpha in [8/5, 3]).
Write p = c(alpha) q; then Phi_q''(alpha) is an explicit expression F(alpha, p)
in alpha, l(p) = log zeta(1+p) and its first two p-derivatives (with
c'/c = 2/(3(alpha-1)), c''/c = -2/(9(alpha-1)^2)).  For p <= 0.9 the script
uses zeta(1+p) = 1/p + D(p), drops the term -alpha log q (linear in alpha),
divides by p > 0, and evaluates the entire function D from its Taylor
polynomial of degree 40 at 0 (coefficients from Arb) with Berndt's bound
|gamma_n| <= 4 (n-1)!/pi^n for the tail; for p >= 0.9 it uses
zeta(1+p) - 1 = zeta(1+p, 2) (Hurwitz).  The rectangle alpha in [8/5, 3],
p in [p_lo, 12.8], p_lo < c(8/5) 1e-40, contains every (alpha, c(alpha) q) with
1e-40 <= q <= 8.  It is cut into boxes along a stored subdivision tree
(convex_tree.json, found by make_convex_tree.py; children share exact binary64
endpoints).  On each leaf the script checks F > 0 by the mean value form
F(box) in F(centre) + dF/dalpha(box) [-r_alpha, r_alpha]
                    + dF/dlog p(box) [-r_v, r_v],
with the gradient enclosed by forward-mode differentiation in ball arithmetic.
Leaves marked as outside are checked to miss the image of q in [1e-40, 8].
The script also certifies the bounds on D used for q < 1e-40.

Usage: python3 unimodality.py   (needs python-flint and numpy)
"""
import json
import math
import sys
import time
from fractions import Fraction as Fr
from pathlib import Path

import numpy as np
from flint import acb, arb, arb_series, ctx

HERE = Path(__file__).resolve().parent
U53 = 2.0 ** -53
TINY = sys.float_info.min


def A(fr):
    fr = Fr(fr)
    return arb(fr.numerator) / fr.denominator


def up(x):
    """Binary64 number >= the arb ball x."""
    return math.nextafter(float(x.upper()), math.inf)


def lo(x):
    """Binary64 number <= the arb ball x."""
    return math.nextafter(float(x.lower()), -math.inf)


def outward_float(value, direction):
    exact = Fr(value)
    near = float(exact)
    if Fr.from_float(near) == exact:
        return near
    return math.nextafter(near, direction)


# ---------------------------------------------------------------- Part 1

def geometric(a, b, r):
    """Increasing binary64 nodes from a with ratio about r, last one >= b."""
    n = int(math.log(b / a) / math.log(r)) + 2
    x = a * r ** np.arange(n)
    x[0] = a
    while x[-1] < b:
        x = np.append(x, x[-1] * r)
    assert np.all(x[1:] > x[:-1])
    return x


def Zball(q, al):
    """Z(q) = (1 + zeta(1+q, 2))^alpha - 1 for an exact q > 0."""
    w = (arb(1) + arb(q)).zeta(2)
    return (al * w.log1p()).expm1()


def upper(alpha, sigma, tau, verbose=True):
    ctx.prec = 128
    al, SIG, TAU = A(alpha), A(sigma), A(tau)
    RHO = (al - 1) / 2
    beta = RHO - SIG
    assert beta > 0 and beta <= 2 and SIG + RHO > 0
    assert TAU > 0 and TAU < arb(2).log()
    assert arb(3).zeta() ** (al - 1) < 2
    xs, XL, y0, Y = 1e-8, 80.0, 1e-13, 40.0

    # Z table and chord slopes
    Q = geometric(0.9e-8, XL + Y + 1, 1.005)
    assert np.all(Q[1:] <= 2 * Q[:-1])
    Zb = [Zball(float(q), al) for q in Q]
    Zup = np.array([up(z) for z in Zb])
    slope = np.array([up((Zb[j] - Zb[j + 1]) / (arb(float(Q[j + 1])) - arb(float(Q[j]))))
                      for j in range(len(Q) - 1)] + [0.0])
    assert np.all(Zup >= TINY) and np.all(slope >= 0)

    def Zchord(q):
        """Binary64 values, each at most (1-u)^-2 times an upper bound of Z(q_true)."""
        ql = np.nextafter(q, -np.inf)
        j = np.searchsorted(Q, ql, side="right") - 1
        assert np.all(j >= 0) and np.all(j < len(Q) - 1)
        d = Q[j + 1] - ql            # exact (Sterbenz)
        t = slope[j] * d
        assert np.all((t == 0) | (t >= TINY))
        return Zup[j + 1] + t

    # y grid and chord weights
    yg = geometric(y0, 2.0, 1.01)
    yg = yg[yg < 2.0]
    yg = np.concatenate([yg, 2.0 + np.arange(0, int((Y - 2) * 50) + 1) / 50])
    assert yg[-1] == Y and np.all(yg[1:] > yg[:-1])
    ctx.prec = 192
    a0, a1 = 1 + beta, 2 + beta
    F0 = [arb(float(y) * TAU).gamma_lower(a0) / TAU ** a0 for y in yg]
    F1 = [arb(float(y) * TAU).gamma_lower(a1) / TAU ** a1 for y in yg]
    W = [arb(0)] * len(yg)
    for k in range(len(yg) - 1):
        ya, yb = arb(float(yg[k])), arb(float(yg[k + 1]))
        h = yb - ya
        I0, I1 = F0[k + 1] - F0[k], F1[k + 1] - F1[k]
        W[k] += (yb * I0 - I1) / h
        W[k + 1] += (I1 - ya * I0) / h
    Wf = np.array([up(w) for w in W])
    assert np.all(Wf >= TINY)
    ctx.prec = 128
    nterms = len(yg)
    sumfac = 1 / (1 - arb(nterms + 3) * arb(U53))
    b = arb(2).log() + TAU
    tailY = 2 * al * (-b * Y).exp() * (arb(Y) ** 2 / b + 2 * arb(Y) / b ** 2 + 2 / b ** 3)
    headf = arb(y0) ** (1 + beta) / (1 + beta) / (1 - arb(U53)) ** 2

    X = geometric(xs, XL, 1.01)
    Jup = []
    for i0 in range(0, len(X), 256):
        xb = X[i0:i0 + 256]
        Zm = Zchord(xb[:, None] + yg[None, :])
        prod = Zm * Wf[None, :]
        assert np.all(np.isfinite(prod)) and np.all(prod >= TINY)
        sums = prod.sum(axis=1)
        heads = Zchord(xb)
        for x, s, hz in zip(xb, sums, heads):
            Jup.append(arb(float(s)) * sumfac + arb(float(hz)) * headf
                       + tailY * arb(2) ** (-arb(float(x))))
    worst, arg = arb(0), None
    sr = SIG + RHO
    for i in range(len(X) - 1):
        x1, x2 = arb(float(X[i])), arb(float(X[i + 1]))
        h = x2 - x1
        J1, J2 = arb(Jup[i].upper()), arb(Jup[i + 1].upper())
        d = sr / x1 + TAU + (J2 / J1 - 1) / h
        dpos = d if d > 0 else (arb(0) if d < 0 else arb(0).union(d))
        B = x1 ** sr * (TAU * x1).exp() * J1 * (dpos * h).exp()
        if B.upper() > worst.upper():
            worst, arg = arb(B.upper()), float(X[i])
    # x <= xs: zeta(1+q) <= (1+q)/q
    ys = arb(1) / 10 ** 4
    xsA = arb(xs)
    Bt = arb.gamma(1 + RHO - SIG) * arb.gamma(RHO + SIG) / arb.gamma(1 + 2 * RHO)
    assert al / TAU >= 1 + xsA
    M = (al / TAU) ** al * (-al + TAU * (1 + xsA)).exp()
    r1 = (TAU * xsA).exp() * ((1 + xsA + ys) ** al * Bt
                              + M * (xsA / ys) ** (RHO + SIG) / (RHO + SIG))
    # x >= XL: Z(q) <= 2 alpha 2^-q, and x^(sigma+rho) e^(-(log2-tau)x) decreasing
    L2 = arb(2).log()
    assert arb(XL) > (SIG + RHO) / (L2 - TAU)
    r2 = 2 * al * arb(XL) ** (SIG + RHO) * (-(L2 - TAU) * XL).exp() \
        * arb.gamma(1 + RHO - SIG) / (TAU + L2) ** (1 + RHO - SIG)
    U = max(worst.upper(), r1.upper(), r2.upper())
    if verbose:
        print(f"  upper a={alpha} sig={sigma} tau={tau}: cells {len(X) - 1}, "
              f"middle {float(worst.upper()):.7f} at x~{arg:.3g}, "
              f"small-x {float(r1.upper()):.5f}, large-x {float(r2.upper()):.2e}")
    return arb(U)


def lower(alpha, s, t):
    ctx.prec = 96
    al, S, T = A(alpha), A(s), A(t)
    assert Fr(alpha) == 3  # Z(q) = w (3 + 3w + w^2), w = zeta(1+q, 2)
    RHO = (al - 1) / 2
    e = al - 2 * S
    assert e > 0 and S < arb(1) / 2

    def f(z, analytic):
        if analytic and not (z.real > 0):
            return acb(float("nan"))
        w = (1 + z).zeta(2)
        return w * (3 + w * (3 + w)) * (e * z.log() - T * z).exp()

    I = acb.integral(f, arb("1e-6"), 60, rel_tol=arb(2) ** -40).real
    pref = arb.gamma(1 + RHO - S) ** 2 / arb.gamma(2 + 2 * RHO - 2 * S) \
        * (2 * T) ** (1 - 2 * S) / arb.gamma(1 - 2 * S)
    return arb((pref * I).lower())


def Lam(alpha):
    ctx.prec = 96
    al = A(alpha)
    return arb.gamma(al / 2) ** 2 / arb.gamma(al)


def part1():
    U19 = upper("19/10", "2734/10000", "5587/10000")
    U26 = upper("13/5", "-2552/10000", "6175/10000")
    L3 = lower("3", "-5302/10000", "6229/10000")
    L16 = arb(Lam("8/5").lower())
    print(f"N(19/10) <= {float(U19.upper()):.7f}   N(13/5) <= {float(U26.upper()):.7f}   "
          f"N(3) >= {float(L3.lower()):.7f}   Lambda(8/5) >= {float(L16.lower()):.7f}")
    assert U19 < A("1.2313") and U26 < A("1.3438")
    assert L3 > A("1.6031") and L16 > A("1.5169")
    # Lemma 6.21: slopes from the rational thresholds printed in the book.
    ctx.prec = 96
    s_dec = A(Fr(10, 3)) * ((A("1.2313") / A("1.9")).log() - (A("1.5169") / A("1.6")).log())
    s_inc = A(Fr(5, 2)) * ((A("1.6031") / A("3")).log() - (A("1.3438") / A("2.6")).log())
    assert s_dec < A("-1.26") and s_inc > 0
    print(f"decreasing slope <= {up(s_dec):.5f} < -1.26; increasing slope >= {lo(s_inc):.5f} > 0")


# ---------------------------------------------------------------- Part 2

class Dual:
    """Value and gradient (d/dalpha, d/dlog p) over arb balls."""
    __slots__ = ("v", "a", "b")

    def __init__(self, v, a, b):
        self.v, self.a, self.b = v, a, b

    def __add__(s, o):
        if isinstance(o, Dual):
            return Dual(s.v + o.v, s.a + o.a, s.b + o.b)
        return Dual(s.v + o, s.a, s.b)
    __radd__ = __add__

    def __neg__(s):
        return Dual(-s.v, -s.a, -s.b)

    def __sub__(s, o):
        return s + (-o)

    def __rsub__(s, o):
        return (-s) + o

    def __mul__(s, o):
        if isinstance(o, Dual):
            return Dual(s.v * o.v, s.a * o.v + s.v * o.a, s.b * o.v + s.v * o.b)
        return Dual(s.v * o, s.a * o, s.b * o)
    __rmul__ = __mul__

    def chain(s, f0, f1):
        return Dual(f0, s.a * f1, s.b * f1)

    def inv(s):
        r = 1 / s.v
        return s.chain(r, -r * r)

    def __truediv__(s, o):
        return s * (o.inv() if isinstance(o, Dual) else 1 / arb(o))

    def __rtruediv__(s, o):
        return s.inv() * o

    def log(s):
        return s.chain(s.v.log(), 1 / s.v)

    def exp(s):
        e = s.v.exp()
        return s.chain(e, e)

    def expm1(s):
        e = s.v.expm1()
        return s.chain(e, e + 1)


NC = 40
DCOEF = None


def dcoef():
    """Taylor coefficients of D(p) = zeta(1+p) - 1/p at 0."""
    global DCOEF
    if DCOEF is None:
        old = ctx.prec
        ctx.prec = 128
        ds = arb_series([arb(1), 1], prec=NC + 1).zeta(1, True)
        DCOEF = [ds[i] for i in range(NC + 1)]
        ctx.prec = old
    return DCOEF


def Dseries(p):
    """Taylor coefficients of order 0..3 of x -> D(p + x), for a ball |p| < 1.
    Tail: |a_n| <= 4/(n pi^n) (Berndt), C(n, j) <= n^j."""
    c = dcoef()
    R = abs(p).upper()
    assert R < 1
    P = arb_series([p, 1], prec=4)
    acc = arb_series([c[NC]], prec=4)
    for n in range(NC - 1, -1, -1):
        acc = acc * P + c[n]
    r = R / arb.pi()
    out = []
    for j in range(4):
        rr = r * (1 + arb(1) / (NC + 1)) ** j
        assert rr < 1
        T = 4 * arb(NC + 1) ** (j - 1) * r ** (NC + 1) / (1 - rr) / R ** j
        out.append(acc[j] + arb(0, T.upper()))
    return arb_series(out, prec=4)


def lderivs(p, deflated):
    """[f, f', f''] as duals: f = log(1 + p D(p)) (deflated) or log zeta(1+p)."""
    if deflated:
        l = (arb_series([p.v, 1], prec=4) * Dseries(p.v) + 1).log()
    else:
        l = (arb_series([1 + p.v, 1], prec=4).zeta(2) + 1).log()
    k = [l[0], l[1], 2 * l[2], 6 * l[3]]
    return [p.chain(k[i], k[i + 1]) for i in range(3)]


def F(al, p, deflated):
    """Phi_q''(alpha) at q = p/c(alpha); divided by p in the deflated form."""
    a1 = al - 1
    r1 = 2 / (3 * a1)          # c'/c
    r2 = -r1 / (3 * a1)        # c''/c
    f0, f1, f2 = lderivs(p, deflated)
    if deflated:               # l = h - log p, h = log(1 + p D(p))
        pl1 = p * f1 - 1
        p2l2 = p * p * f2 + 1
        l0 = f0 - p.log()
        base = 2 * f1 * r1 + al * f2 * p * r1 * r1 + al * f1 * r2   # (alpha h(p))''/p
    else:
        pl1 = p * f1
        p2l2 = p * p * f2
        l0 = f0
        base = 2 * (al - 2) / (3 * a1 * a1)                          # (alpha log c)''
    # y = alpha l(p): y' = l + alpha p l' c'/c, y'' = 2 p l' c'/c + alpha p^2 l'' (c'/c)^2 + alpha p l' c''/c
    y1 = l0 + al * pl1 * r1
    y2 = 2 * pl1 * r1 + al * p2l2 * r1 * r1 + al * pl1 * r2
    if deflated:
        # log(1 - e^-y) with e^-y = X = (p/H)^alpha; T' = X/(1-X), T'' = -X/(1-X)^2
        Xp = ((al - 1) * p.log() - al * f0).exp()     # X/p
        X = Xp * p
        T1 = Xp / (1 - X)
        T2 = -T1 / (1 - X)
        return base + T1 * y2 + T2 * y1 * y1
    # log Z = y + log(1 - e^-y); T' = 1/(e^y - 1), T'' = -T'(1 + T')
    T1 = 1 / (al * l0).expm1()
    T2 = -T1 * (1 + T1)
    return base + (1 + T1) * y2 + T2 * y1 * y1


QMIN, QMAX = A("1e-40"), arb(8)


def cfun(a):
    return ((arb(a) - 1).log() * 2 / 3).exp()


def outside(a1, a2, p1, p2):
    return arb(p1) > QMAX * cfun(a2) or arb(p2) < QMIN * cfun(a1)


def box(a1, a2, p1, p2, deflated):
    """Enclosure of F on [a1,a2] x [p1,p2] by the mean value form, and the two
    gradient terms (None if the centre value is not positive)."""
    Ab = arb(a1).union(arb(a2))
    Pb = arb(p1).union(arb(p2))
    V = arb(p1).log().union(arb(p2).log())
    z = arb(0)
    v0 = F(Dual(arb(Ab.mid()), z, z), Dual(arb(V.mid()).exp(), z, z), deflated).v
    if not v0 > 0:
        return v0, None, None
    g = F(Dual(Ab, arb(1), z), Dual(Pb, z, Pb), deflated)
    ea, eb = g.a * arb(0, Ab.rad()), g.b * arb(0, V.rad())
    return v0 + ea + eb, ea, eb


def children(node, flag):
    a1, a2, p1, p2 = node
    if flag == "A":
        m = (a1 + a2) / 2
        assert a1 < m < a2
        return [(a1, m, p1, p2), (m, a2, p1, p2)]
    m = math.sqrt(p1 * p2)
    assert p1 < m < p2
    return [(a1, a2, p1, m), (a1, a2, m, p2)]


def ranges():
    A0 = outward_float(Fr(8, 5), -math.inf)
    return A0, 3.0, [(7e-41, 0.9, True), (0.9, 2.5, False), (2.5, 12.8, False)]


def check_tree(tree):
    ctx.prec = 64
    A0, A1, rngs = ranges()
    assert arb(A0) < A(Fr(8, 5))
    assert arb(rngs[0][0]) < QMIN * cfun(A(Fr(8, 5)))
    assert arb(rngs[-1][1]) > QMAX * cfun(3)
    assert all(rngs[i][1] == rngs[i + 1][0] for i in range(len(rngs) - 1))
    nleaf = nout = 0
    worst = None
    for (p0, p1, deflated), flags in zip(rngs, tree):
        stack = [(A0, A1, p0, p1)]
        pos = 0
        while stack:
            node = stack.pop()
            fl = flags[pos]
            pos += 1
            if fl == "L":
                v = box(*node, deflated)[0]
                assert v > 0, node
                nleaf += 1
            elif fl == "O":
                assert outside(*node), node
                nout += 1
            else:
                stack += children(node, fl)
        assert pos == len(flags)
    return nleaf, nout


def assert_small_q_bounds():
    """Taylor/Berndt certificate for the analytic q < 1e-40 argument."""
    ctx.prec = 128
    # c(alpha) < 2 on [8/5, 3], hence 0 <= p = cq <= 2e-40.
    p = A(0).union(A("2e-40"))
    ds = Dseries(p)
    assert ds[0] > A("0.57") and ds[0] < 1
    assert ds[1] > 0 and ds[1] < 1
    assert -1 < 2 * ds[2] < 1


def part2():
    t = time.time()
    assert_small_q_bounds()
    tree = json.loads((HERE / "convex_tree.json").read_text())["flags"]
    nleaf, nout = check_tree(tree)
    print(f"convex: Phi_q''(alpha) > 0 on alpha in [8/5,3], q in [1e-40,8]: "
          f"{nleaf} boxes, {nout} outside, {time.time() - t:.1f}s")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    t0 = time.time()
    if what in ("all", "bounds"):
        part1()
    if what in ("all", "convex"):
        part2()
    print(f"ALL CHECKS PASSED ({time.time() - t0:.1f}s)")
