"""Exact check of the finite-jet identities in Lemma 7 of
"Remarks on three sharpness questions for composition operators on spaces of
Dirichlet series".

For a1, a2 > 0 and real x1, x2, g consider
    p(z1,z2) = sum_j [ a_j (1-z_j) + (a_j/2 - a_j^2 + i x_j)(1-z_j)^2 ]
               + (-2 a1 a2 + i g)(1-z1)(1-z2),
put z1 = exp(i a2 (u+v)), z2 = exp(i a1 (u-v)), G = Re p, H = Im p.
The curve u = A(v) solves dG/du = 0, f1(v) = G(A(v),v), f3(v) = H(A(v),v).
The script computes the Taylor coefficients [f1]_3, [f1]_4, [f3]_2, [f3]_3,
[f3]_4 in the field Q(a1,a2,x1,x2,g) and checks every identity displayed in
the proof of the lemma.  All arithmetic is exact (rational functions over Q);
reductions modulo the relation S^3 g^2 + W = 0 are done by polynomial
remainder, so no square roots or floating point occur.

Run: python3 jets.py     (needs sympy; tested with sympy 1.13.3; about one second)
Prints "ALL OK" on success.
"""
import math
import sympy as s
from sympy import QQ

F, A1, A2, X1, X2, Gm = s.field('a1 a2 x1 x2 g', QQ)
R, u, v = s.ring('u v', F)
N = 6  # total degree kept in (u, v); enough for the v^4 coefficients


def tr(p):
    return R({m: c for m, c in p.terms() if m[0] + m[1] <= N})


def one_minus_exp(t):
    """Real and imaginary parts of 1 - e^{it}, truncated: (1 - cos t, -sin t)."""
    c, sn, pw = R(0), R(0), R(1)
    for k in range(1, N + 1):
        pw = tr(pw * t)
        f = QQ(1, math.factorial(k))
        if k % 2 == 0:
            c += -(-1) ** (k // 2) * f * pw
        else:
            sn += (-1) ** ((k - 1) // 2) * f * pw
    return c, -sn


def mul(p, q):
    return (tr(p[0] * q[0] - p[1] * q[1]), tr(p[0] * q[1] + p[1] * q[0]))


def scale(z, p):
    return (z[0] * p[0] - z[1] * p[1], z[0] * p[1] + z[1] * p[0])


D1 = one_minus_exp(A2 * (u + v))
D2 = one_minus_exp(A1 * (u - v))
terms = [scale((A1, 0), D1), scale((A2, 0), D2),
         scale((A1 / 2 - A1 * A1, X1), mul(D1, D1)),
         scale((A2 / 2 - A2 * A2, X2), mul(D2, D2)),
         scale((-2 * A1 * A2, Gm), mul(D1, D2))]
G = sum(t[0] for t in terms)
H = sum(t[1] for t in terms)
# leading terms: G = 4 a1^2 a2^2 u^2 + O(3), H = -2 a1 a2 u + O(2)
assert {m: c for m, c in G.terms() if sum(m) <= 2} == {(2, 0): 4 * A1**2 * A2**2}
assert {m: c for m, c in H.terms() if sum(m) <= 1} == {(1, 0): -2 * A1 * A2}


def along(p, A, n):
    """Coefficients of p(A(v), v) up to v^n, for a power series A with A(0)=0."""
    res = [F(0)] * (n + 1)
    pows = [[F(1)] + [F(0)] * n]
    for _ in range(N):
        new = [F(0)] * (n + 1)
        for k1, e1 in enumerate(pows[-1]):
            for k2, e2 in enumerate(A):
                if e1 and e2 and k1 + k2 <= n:
                    new[k1 + k2] += e1 * e2
        pows.append(new)
    for (i, j), c in p.terms():
        for k, ck in enumerate(pows[i]):
            if ck and k + j <= n:
                res[k + j] += c * ck
    return res


n = 4
A = [F(0)] * (n + 1)          # A(v) = sum A[k] v^k, A[0] = A[1] = 0
Gu = G.diff(u)
for k in range(2, n + 1):     # solve dG/du(A(v), v) = 0 order by order
    A[k] = -along(Gu, A, k)[k] / (8 * A1**2 * A2**2)
assert all(c == 0 for c in along(Gu, A, n))
f1 = along(G, A, n)
f3 = along(H, A, n)
assert f1[:3] == [0, 0, 0] and f3[:2] == [0, 0]

a1, a2, x1, x2, g = s.symbols('a1 a2 x1 x2 g')
ex = lambda q: s.sympify(str(q.as_expr()))
F13, F14, F32, F33, F34 = ex(f1[3]), ex(f1[4]), ex(f3[2]), ex(f3[3]), ex(f3[4])
S = a1 + a2
zero = lambda e: s.cancel(s.together(e)) == 0

# displayed formulas for [f1]_3 and 8 a1 a2 [f3]_2
assert zero(F13 - (a2**3 * x1 - a1**3 * x2 + a1 * a2 * (a1 - a2) * g / 2))
assert zero(8 * a1 * a2 * F32 - (a2**3 * (6 - 8 * a1) * x1 + a1**3 * (6 - 8 * a2) * x2
                                 + a1 * a2 * (8 * a1 * a2 - S) * g))
# determinant of the linear system in (x1, x2)
M = s.Matrix([[s.diff(F13, x1), s.diff(F13, x2)], [s.diff(F32, x1), s.diff(F32, x2)]])
assert zero(M.det() + a1**2 * a2**2 * (2 * S - 3) / 2)
# its solution
xs = a1 * (2 * a1 * a2 + a1 + 2 * a2**2 - 2 * a2) / (2 * a2**2 * (2 * S - 3)) * g
ys = a2 * (2 * a1**2 + 2 * a1 * a2 - 2 * a1 + a2) / (2 * a1**2 * (2 * S - 3)) * g
sub = {x1: xs, x2: ys}
assert zero(F13.subs(sub)) and zero(F32.subs(sub))
# [f1]_4 on that solution
W = a1 * a2 * (2 * S - 3)**2 * (a1 * a2 * S - a1**2 - a2**2 + a1 * a2)
assert zero(F14.subs(sub) + S * (S**3 * g**2 + W) / (4 * (2 * S - 3)**2))


def reduces_to(expr, target, rel):
    """expr == target modulo the polynomial relation rel(g) = 0 (rel monic up to a
    nonzero factor free of g): the numerator of expr - target is divisible by rel."""
    num, den = s.fraction(s.cancel(s.together(expr - target)))
    return s.rem(s.Poly(num, g), s.Poly(rel, g)).is_zero


rel = S**3 * g**2 + W
# [f3]_3 = (a1 - a2) P / (4S) modulo [f1]_4 = 0
P = 2 * a1**3 + 2 * a2**3 + a1 * a2**2 + a1**2 * a2 - 3 * a1**2 - 3 * a2**2 + 3 * a1 * a2
assert reduces_to(F33.subs(sub), (a1 - a2) * P / (4 * S), rel)
Dd, Ss = s.symbols('D S')
assert s.expand((4 * P).subs({a1: (Ss + Dd) / 2, a2: (Ss - Dd) / 2})
                + 3 * Ss**2 * (1 - Ss) + (9 - 5 * Ss) * Dd**2) == 0
# equal case a1 = a2 = a: g^2 = -a(2a-1)(4a-3)^2/8 and [f3]_4 = 3a^2(2a-1) g / (8(4a-3))
a = s.symbols('a')
eq = {a1: a, a2: a}
rel_eq = s.expand(rel.subs(eq))
assert zero(rel_eq / (8 * a**3) - (g**2 + a * (2 * a - 1) * (4 * a - 3)**2 / 8))
assert reduces_to(F34.subs(sub).subs(eq), 3 * a**2 * (2 * a - 1) * g / (8 * (4 * a - 3)), rel_eq)
print("ALL OK")
