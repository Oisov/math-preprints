"""Certificates for Section 8.2 (depth at p=4) and Theorem 4.

1. Exact Sturm certificate: the sextic P(x) has no real zeros and P(0)>0, so
   4C(theta)-B(theta)^2>0, i.e. Re Omega > c |Im Omega| (Section 8.2).
2. Numerical checks (not part of any proof): the identity 4C-B^2 = P/(...),
   the closed form W_4(C) = 2 - C/2 - conj(C) + |C|^2/2, and
   d/dp Im W_p(2-2i) at p=2 equal to -(pi+2 log 2)/16 (Theorem 4).
Standard library for part 1; numpy/scipy for part 2.
"""
from fractions import Fraction as F
from functools import reduce
from math import gcd

P = list(map(F, [6161460000, -11071132600, 8715093039, -8819558000,
                 3250290474, 1699867400, 160903671]))


def trim(a):
    while len(a) > 1 and a[-1] == 0:
        a.pop()
    return a


def primitive(a):
    scale = 1
    for x in a:
        scale = scale * x.denominator // gcd(scale, x.denominator)
    ints = [int(x * scale) for x in a]
    d = reduce(gcd, ints, 0)
    return [F(x // d) for x in ints]


def remainder(a, b):
    a = a[:]
    while len(a) >= len(b) and any(a):
        k = len(a) - len(b)
        q = a[-1] / b[-1]
        for j, v in enumerate(b):
            a[j + k] -= q * v
        trim(a)
    return a


def sturm_certificate():
    S = [primitive(P), primitive([i * P[i] for i in range(1, len(P))])]
    while len(S[-1]) > 1:
        S.append(primitive([-v for v in remainder(S[-2], S[-1])]))

    def signs(neg):
        return [1 if (a[-1] * ((-1) ** (len(a) - 1) if neg else 1)) > 0 else -1
                for a in S]

    def var(s):
        return sum(x != y for x, y in zip(s, s[1:]))
    assert var(signs(True)) == var(signs(False)) == 3
    assert P[0] > 0
    print("sturm: P has no real zeros and P(0) > 0  [exact]")


def numerical_checks():
    import numpy as np
    from scipy.integrate import quad
    b, d, c = 887 / 1250, -27 / 625, 473 / 200
    for th in (0.3, -0.7, 1.2, -1.45):
        B = -3 * np.cos(th) - c * np.sin(th) + d * (np.cos(3 * th) + c * np.sin(3 * th))
        C = 2 + b * (np.cos(2 * th) + c * np.sin(2 * th))
        x = np.tan(th)
        Pv = sum(float(P[i]) * x ** i for i in range(7))
        assert abs((4 * C - B * B) - Pv / (3906250000 * (1 + x * x) ** 3)) < 1e-9

    def W(p, C):
        def part(t, f):
            H = 1 / (1 - 1j * t)
            E = 1 - C * H
            return f((1 - abs(E) ** (p - 2) * E) * np.conj(H) / (np.pi * C))
        re = quad(lambda t: part(t, np.real), -np.inf, np.inf, limit=400)[0]
        im = quad(lambda t: part(t, np.imag), -np.inf, np.inf, limit=400)[0]
        return re + 1j * im
    for C in (2 - 2j, (3 - 1j * np.sqrt(7)) / 2):
        closed = 2 - C / 2 - np.conj(C) + abs(C) ** 2 / 2
        assert abs(W(4, C) - closed) < 1e-7
    h = 1e-4
    der = (W(2 + h, 2 - 2j).imag - W(2 - h, 2 - 2j).imag) / (2 * h)
    kappa = (np.pi + 2 * np.log(2)) / 16
    assert abs(der + kappa) < 1e-6
    C = (3 - 1j * np.sqrt(7)) / 2
    w = W(4, C)
    print(f"numerics: W_4 closed form ok; dImW/dp at 2 = {der:.9f} (-kappa = {-kappa:.9f});"
          f" Re W/|W| at p=4 = {w.real/abs(w):.9f} (sqrt(7/8) = {np.sqrt(7/8):.9f})")


def main():
    sturm_certificate()
    numerical_checks()
    return True


if __name__ == "__main__":
    main()
