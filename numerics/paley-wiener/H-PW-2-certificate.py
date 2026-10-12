#!/usr/bin/env python3
"""Interval-arithmetic certificate for Conjecture 9.2 of Brevig--Chirre--
Ortega-Cerda--Seip, "Point evaluation in Paley--Wiener spaces"
(arXiv:2210.13922):

    C_p >= (p/2) * (1 - A (p - 2))   for all 1 <= p <= 2,

with the explicit constant A = 7/100 (default) or A = 3/50.

Requires python-flint 0.9.0 (Arb ball arithmetic).  Usage:
    python H-PW-2-certificate.py          # A = 7/100
    python H-PW-2-certificate.py 0.06     # A = 3/50

Method (the proofs of every step are in the accompanying paper):

* Test functions g_al(z) = Gamma(al)^2 / (Gamma(al-z) Gamma(al+z)),
  al = 1/2 + 1/q, which lie in PW^p for p > q/2 and satisfy g_al(0) = 1.
  Hence C_p >= 1 / J_q(p), J_q(p) = int_R |g_al|^p.
* p -> log J_q(p) is convex (Hoelder).  On a patch [a, b] with a fixed q,
  log J_q(p) lies below the chord through (a, log U_a), (b, log U_b) where
  U_a >= J_q(a), U_b >= J_q(b) are rigorous upper bounds.
* H_A(p) = log(2 / (p (1 + A (2 - p)))) is convex, so Delta = H_A - chord is
  convex; Delta >= 0 on [a, b] is checked by tangent-line lower bounds on
  a subdivision of [a, b].
* Last patch [a, 2] uses sinc (q = 2) with J_2(2) = 1 exactly (Plancherel),
  so Delta(2) = 0, and Delta >= 0 on [a, 2] follows from Delta'(2) <= 0.
* J_q(p) upper bounds: rigorous Arb integration (acb.integral) on the
  zero-free cells [0, al], [al + n - 1, al + n], 1 <= n < N, plus an explicit
  tail majorant for n >= N.
"""

import sys
import time

from flint import acb, arb, ctx, fmpq

ctx.prec = 80
NAN = acb(arb.nan())


def as_arb(x):
    return arb(x) if not isinstance(x, fmpq) else arb(x.p) / arb(x.q)


def J_bounds(q, p, N=400, rel_tol=2.0**-40):
    """Return (lower, upper) arb numbers with lower <= J_q(p) <= upper.

    J_q(p) = int_R |g_al(x)|^p dx, al = 1/2 + 1/q, s = 2al - 1 = 2/q.
    """
    q = as_arb(q)
    p = as_arb(p)
    al = arb(1) / 2 + 1 / q
    s = 2 / q
    sp = s * p
    if not sp > 1:
        raise ValueError("g_al not in L^p: need p > q/2")
    G2 = al.gamma() ** 2
    K = G2 / arb.pi()
    lK = K.log()

    def powabs(u, analytic, pre):
        # exp(p (pre + log u)): the analytic continuation of
        # (|e^pre| |u|)^p from the real segment where u > 0.
        v = (p * (pre + u.log(analytic=analytic))).exp()
        if v.is_finite() or analytic:
            return v
        # Non-analytic call on a ball containing a zero of u: crude enclosure
        # |f| <= |exp(p pre)| * |u|^p (valid since u is real >= 0 on the path).
        M = (p * pre).exp().abs_upper() * (u.abs_upper() ** p)
        return acb(arb(0, M), arb(0, M))

    # inner cell [0, al]:  g = G2 * rgamma(al - x) * rgamma(al + x) >= 0.
    # Substitute x = al * u (u in [0, 1]) so that the endpoints are exact;
    # the factor al (dx = al du) is put into the log-prefactor.
    lal = al.log()

    def f_inner(u, analytic):
        x = al * u
        pre = (G2 * (al + x).rgamma()).log(analytic=analytic)
        if not pre.is_finite():
            return NAN
        return powabs((al - x).rgamma(), analytic, pre + lal / p)

    total = acb.integral(f_inner, 0, 1, rel_tol=arb(rel_tol)).real

    # cells x = al + n - 1 + t, t in [0, 1], n >= 1:
    # |g(x)| = K * Gamma(n + t) / Gamma(n + t + s) * sin(pi t)   (by (7.3))
    for n in range(1, N):
        nn = acb(n)

        def f_cell(t, analytic, nn=nn):
            z = nn + t
            if analytic and not (z.real > 0):
                return NAN  # lgamma branch cut must be avoided
            lr = lK + z.lgamma() - (z + s).lgamma()
            if not lr.is_finite():
                return NAN
            return powabs((t * acb.pi()).sin(), analytic, lr)

        total += acb.integral(f_cell, 0, 1, rel_tol=arb(rel_tol)).real

    # tail n >= N:  int_cell <= K^p (Gamma(n)/Gamma(n+s))^p m_p
    #               <= K^p m_p n^{-sp},
    #   m_p = int_0^1 sin(pi t)^p dt = Gamma((p+1)/2) / (sqrt(pi) Gamma(p/2+1)),
    #   sum_{n>=N} n^{-sp} <= N^{-sp} + N^{1-sp}/(sp-1).
    mp = ((p + 1) / 2).gamma() / (arb.pi().sqrt() * (p / 2 + 1).gamma())
    Nn = arb(N)
    tail = K ** p * mp * (Nn ** (-sp) + Nn ** (1 - sp) / (sp - 1))
    lower = 2 * total
    upper = 2 * (total + tail)
    return arb(lower.lower()), arb(upper.upper())


def H(A, p):
    return (arb(2) / (p * (1 + A * (2 - p)))).log()


def dH(A, p):
    return -1 / p + A / (1 + A * (2 - p))


def check_patch(A, a, b, la, lb, pieces=64, depth=12):
    """Certify H_A(p) - chord(p) >= 0 on [a, b] (chord through (a, la), (b, lb)).

    Delta is convex, so on [c, d]:
      Delta(p) >= Delta(c) + min(0, Delta'(c)) (d - c).
    Returns the certified lower bound of Delta on [a, b] (an arb lower end).
    """
    slope = (lb - la) / as_arb(b - a)
    aa = as_arb(a)

    def Delta(p):
        return H(A, p) - (la + slope * (p - aa))

    def dDelta(p):
        return dH(A, p) - slope

    worst = None
    stack = [(a + (b - a) * fmpq(k, pieces), a + (b - a) * fmpq(k + 1, pieces), 0)
             for k in range(pieces)]
    while stack:
        c, d, lev = stack.pop()
        ca, da = as_arb(c), as_arb(d)
        D0 = Delta(ca)
        d0 = dDelta(ca)
        # rigorous: Delta(p) >= Delta(c) + min(0, Delta'(c)) (d - c) on [c, d]
        dl = arb(d0.lower())
        if dl > 0:
            dl = arb(0)
        bound = arb(D0.lower()) + dl * (da - ca)
        lowb = arb(bound.lower())
        if not lowb > 0 and lev < depth:
            m = (c + d) / 2
            stack += [(c, m, lev + 1), (m, d, lev + 1)]
            continue
        if worst is None or lowb < worst:
            worst = lowb
    return worst


# Patches (q, a, b) for A = 3/50 and A = 7/100; designed numerically, certified
# below.  The final patch [a_last, 2] uses sinc (q = 2).
PATCHES = {
    fmpq(3, 50): [
        (fmpq(447, 400), fmpq(1), fmpq(124, 100)),
        (fmpq(1415, 1000), fmpq(124, 100), fmpq(597, 400)),
        (fmpq(1605, 1000), fmpq(597, 400), fmpq(687, 400)),
        (fmpq(1775, 1000), fmpq(687, 400), fmpq(745, 400)),
        (fmpq(753, 400), fmpq(745, 400), fmpq(194, 100)),
        (fmpq(194, 100), fmpq(194, 100), fmpq(779, 400)),
    ],
    fmpq(7, 100): [
        (fmpq(1122, 1000), fmpq(1000, 1000), fmpq(1182, 1000)),
        (fmpq(1314, 1000), fmpq(1182, 1000), fmpq(1370, 1000)),
        (fmpq(1471, 1000), fmpq(1370, 1000), fmpq(1548, 1000)),
        (fmpq(1655, 1000), fmpq(1548, 1000), fmpq(1713, 1000)),
        (fmpq(1801, 1000), fmpq(1713, 1000), fmpq(1847, 1000)),
        (fmpq(1912, 1000), fmpq(1847, 1000), fmpq(1941, 1000)),
        (fmpq(1953, 1000), fmpq(1941, 1000), fmpq(1977, 1000)),
    ],
}
SINC_FROM = {fmpq(3, 50): fmpq(779, 400), fmpq(7, 100): fmpq(1977, 1000)}


def main():
    t0 = time.time()
    A = fmpq(7, 100)
    if len(sys.argv) > 1:
        from fractions import Fraction
        fr = Fraction(sys.argv[1]).limit_denominator(10**6)
        A = fmpq(fr.numerator, fr.denominator)
    if A not in PATCHES:
        raise SystemExit("no patch table for A = %s" % A)
    Aa = as_arb(A)
    patches = PATCHES[A]
    a_last = SINC_FROM[A]
    print("Target: C_p >= (p/2)(1 - A(p-2)) on [1,2] with A = %s" % A)
    print("arb precision %d bits" % ctx.prec)

    # sanity: sinc at p = 2 has J = 1 exactly
    lo, up = J_bounds(2, 2)
    print("sanity  J_sinc(2) in [%s, %s]  (exact value 1)" % (lo.str(10), up.str(10)))
    assert lo <= 1 <= up

    ok = True
    covered = fmpq(1)
    print("%-8s %-17s %-24s %-24s %s" % ("q", "[a, b]", "log J_q(a) <=", "log J_q(b) <=",
                                         "min Delta >="))
    for (q, a, b) in patches:
        assert a <= covered, "gap in coverage"
        covered = max(covered, b)
        la_lo, la_up = J_bounds(q, a)
        lb_lo, lb_up = J_bounds(q, b)
        la = arb(la_up.log().upper())
        lb = arb(lb_up.log().upper())
        w = check_patch(Aa, a, b, la, lb)
        good = w > 0
        ok &= bool(good)
        print("%-8s [%-6s, %-6s] %-24s %-24s %s %s" % (
            float(as_arb(q).mid()), float(as_arb(a).mid()), float(as_arb(b).mid()),
            la.str(12), lb.str(12), w.str(5), "ok" if good else "FAIL"))
        print("         (J lower/upper at a: %s / %s; at b: %s / %s)" % (
            la_lo.str(12), la_up.str(12), lb_lo.str(12), lb_up.str(12)), flush=True)

    # final sinc patch [a_last, 2]: Delta(2) = 0, need Delta'(2) <= 0
    assert a_last <= covered
    ja_lo, ja_up = J_bounds(2, a_last, N=2000)
    la = arb(ja_up.log().upper())
    slope = (0 - la) / (2 - as_arb(a_last))
    d2 = dH(Aa, arb(2)) - slope
    good = d2 < 0
    ok &= bool(good)
    print("sinc     [%-6s, 2     ] log J_2(a) <= %s ; Delta(2) = 0, Delta'(2) <= %s %s" % (
        float(as_arb(a_last).mid()), la.str(12), arb(d2.upper()).str(5),
        "ok" if good else "FAIL"))
    print("         (J_2(a) lower/upper: %s / %s)" % (ja_lo.str(12), ja_up.str(12)))
    covered = fmpq(2)

    dt = time.time() - t0
    if ok and covered == 2:
        print("PASS: certified C_p >= (p/2)(1 - A(p-2)) for all p in [1,2] with A = %s "
              "(runtime %.1f s)" % (A, dt))
    else:
        print("FAIL (runtime %.1f s)" % dt)
        sys.exit(1)


if __name__ == "__main__":
    main()
