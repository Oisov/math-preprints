"""Symbolic identities and sign checks used in the book (requires sympy).

1. Thm 8.6 (generator inequality): the curvature formula for -t f''/f', the
   positivity of y/(2(1+y)) - (-t f''/f'), and the coefficient positivity of
   f''' after M = 4 + m.
2. Thm 8.6, case y <= 4/3: the expansion of M^5 times the fourth-power
   difference in powers of M.
3. Thm 9.2 (column estimate): the cubic obtained from the rational bound for
   arctan(u)/u, its concavity and its endpoint signs for 0.8325 < A < 0.8326.
"""
import sympy as sp

c, M, y, m, t, A = sp.symbols("c M y m t A", positive=True)


def check_envelope():
    S4 = 1 + 4 * c**2 + M * c**4
    tt = ((2 * c + M * c**3) / (1 + 2 * c**2)) ** 2
    S = S4 ** sp.Rational(1, 4)
    f = S**3 / (1 + 2 * c**2)
    d = lambda e: sp.diff(e, c) / sp.diff(tt, c)
    fp = d(f)
    assert sp.simplify(tt * fp - c**2 * (2 + M * c**2) / (2 * (1 + 2 * c**2) * S)) == 0
    kappa = sp.simplify(-tt * d(fp) / fp)
    claimed = c**2 * (2 * M**2 * c**6 + 3 * M**2 * c**4 + 12 * M * c**2 + 2 * M - 24 * c**2 - 4) / (
        2 * S4 * (2 * M * c**4 + 3 * M * c**2 - 4 * c**2 + 2))
    assert sp.simplify(kappa - claimed) == 0
    gap = sp.factor(sp.simplify(M * c**2 / (2 * (1 + M * c**2)) - kappa))
    num = sp.numer(sp.together(gap))
    target = c**2 * (2 * M**2 * c**6 + M**2 * c**4 + 8 * M * c**4 + M * c**2 * (M - 4) + 24 * c**2 + 4)
    assert sp.simplify(num / target).is_number
    f3 = sp.together(d(d(fp)))
    core = [a for a in sp.Mul.make_args(sp.factor(sp.numer(f3))) if a.has(M) and a.is_Add][0]
    poly = sp.Poly(sp.expand(core.subs(M, 4 + m)), m, c)
    assert all(v >= 0 for _, v in poly.terms()), "negative coefficient in f'''"
    print("ok  envelope: curvature formula, curvature bound, f''' > 0")


def check_brackets():
    B = sp.expand((M + 2 * y) ** 4 * (M + 4 * y + y**2) - M * (M + 3 * y + y**2 / 4) ** 4)
    claimed = (y**2 * (y + 4) * (4 - 3 * y) / 8 * M**3 + y**3 * (y + 4) * (80 - 32 * y - y**2) / 16 * M**2
               + y**4 * (y + 4) * (4032 - 688 * y - 44 * y**2 - y**3) / 256 * M + 16 * y**5 * (y + 4))
    assert sp.expand(B - claimed) == 0
    for br in [(4 - 3 * y), (80 - 32 * y - y**2), (4032 - 688 * y - 44 * y**2 - y**3)]:
        assert br.subs(y, sp.Rational(4, 3)) >= 0 and br.subs(y, 0) > 0
    print("ok  envelope brackets: identity and signs on 0 < y <= 4/3")


def check_column_estimate():
    R = (105 + 55 * t) / (105 + 90 * t + 9 * t**2)
    Psi = 2 * A * R - 1 / (2 * A * (1 + t)) - 1
    num = sp.expand(sp.together(Psi * 2 * A * (1 + t) * (105 + 90 * t + 9 * t**2)))
    P = ((420 * A**2 - 210 * A - 105) + (640 * A**2 - 390 * A - 90) * t
         + (220 * A**2 - 198 * A - 9) * t**2 - 18 * A * t**3)
    assert sp.expand(num - P) == 0
    lo, hi = sp.Rational(8325, 10000), sp.Rational(8326, 10000)
    for a in (lo, hi):
        assert P.subs({A: a, t: 0}) > 0 and P.subs({A: a, t: 1}) > 0
        assert (220 * a**2 - 198 * a - 9) < 0
    assert lo**2 < sp.log(2) < hi**2
    # the rational bound itself: arctan(u)/u >= R(u^2) on (0,1], checked on a grid
    for k in range(1, 1001):
        u = sp.Rational(k, 1000)
        assert sp.atan(u) / u - R.subs(t, u**2) >= -sp.Rational(1, 10**30)
    print("ok  column estimate: cubic identity, concavity and endpoint signs")


if __name__ == "__main__":
    check_envelope()
    check_brackets()
    check_column_estimate()
    print("ALL OK")
