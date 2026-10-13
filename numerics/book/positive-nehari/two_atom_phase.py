"""Certify the phase integral for the two-atom positive Nehari counterexample.

Run: uv run --with numpy==2.5.4 --with python-flint==0.9.0 python -I \
     two_atom_phase.py

Arb encloses the 256 one-dimensional center kernels and cell denominator
bounds. All bulk arithmetic uses binary64 interval endpoints, with each
basic operation followed by nextafter outwards. Square root is an IEEE-754
correctly rounded operation. No NumPy trigonometric function is used.
Cell lower bounds are rounded DOWN to multiples of 2^-40 and summed as
integers (each plane fits int64, the outer sum uses Python integers).
The exact final comparison is integral Re(conj(U)/U) > 17/100 > 1/A.
See the proof of the two-atom proposition (Positive Helson chapter of the book;
Lemma 6 of "Positive Helson matrices without bounded symbols") for the cell
derivative bound.
"""
import math
from fractions import Fraction

import numpy as np
from flint import acb, arb, ctx

ctx.prec = 128
L = 256
SCALE = 2**40
DOWN, UP = -np.inf, np.inf
np.seterr(all="raise")


def down(x):
    return np.nextafter(x, DOWN)


def up(x):
    return np.nextafter(x, UP)


def arb_lo(x):
    return math.nextafter(float(x.lower()), -math.inf)


def arb_hi(x):
    return math.nextafter(float(x.upper()), math.inf)


def add(a, b):
    return down(a[0] + b[0]), up(a[1] + b[1])


def sub(a, b):
    return down(a[0] - b[1]), up(a[1] - b[0])


def mul(a, b):
    products = [a[i] * b[j] for i in (0, 1) for j in (0, 1)]
    return down(np.minimum.reduce(products)), up(np.maximum.reduce(products))


def square(a):
    ends = [a[0] * a[0], a[1] * a[1]]
    lower = np.where((a[0] <= 0) & (a[1] >= 0), 0, down(np.minimum(*ends)))
    return lower, up(np.maximum(*ends))


def cmul(a, b):
    return sub(mul(a[0], b[0]), mul(a[1], b[1])), add(mul(a[0], b[1]), mul(a[1], b[0]))


def cadd(a, b):
    return add(a[0], b[0]), add(a[1], b[1])


def slice_kernel(table, index):
    return tuple(tuple(component[index] for component in part) for part in table)


def positive_mul_upper(a, b):
    return up(a * b)


def table(sign):
    r, h = arb(4) / 5, arb.pi() / L
    kernels, distance = [], []
    for j in range(L):
        z = (acb(0, 1) * arb.pi() * (2 * j + 1) / L).exp()
        denominator = 1 - sign * r * z
        kernel = 1 / denominator
        kernels.append(((arb_lo(kernel.real), arb_hi(kernel.real)),
                        (arb_lo(kernel.imag), arb_hi(kernel.imag))))
        # |1-sign*r*e^(i theta)| >= max(1-r, |denominator(center)|-r*h).
        distance.append(max(arb_lo(1 - r), arb_lo(abs(denominator) - r * h)))
    data = np.array(kernels)
    assert np.all(np.isfinite(data)) and min(distance) > 0
    # Positive reciprocal upper bounds suffice for all cell derivatives.
    reciprocal = up(1 / np.array(distance))
    return ((data[:, 0, 0], data[:, 0, 1]),
            (data[:, 1, 0], data[:, 1, 1])), reciprocal


def main():
    plus, ip = table(1)
    minus, im = table(-1)
    hp = arb_hi(arb.pi() / L * (arb(4) / 5))
    total = 0
    for j in range(L):
        U = cadd(cmul(cmul(slice_kernel(plus, j), slice_kernel(plus, (slice(None), None))),
                      slice_kernel(plus, (None, slice(None)))),
                 cmul(cmul(slice_kernel(minus, j), slice_kernel(minus, (slice(None), None))),
                      slice_kernel(minus, (None, slice(None)))))
        x2, y2 = square(U[0]), square(U[1])
        norm2 = add(x2, y2)
        assert np.all(norm2[0] > 0)  # center disks may cross zero; centers do not.
        center_abs_lower = down(np.sqrt(norm2[0]))
        phase_numerator = sub(x2, y2)[0]
        phase_lower = down(phase_numerator / np.where(phase_numerator >= 0, norm2[1], norm2[0]))
        P = positive_mul_upper(positive_mul_upper(ip[j], ip[:, None]), ip[None, :])
        N = positive_mul_upper(positive_mul_upper(im[j], im[:, None]), im[None, :])
        ps = up(up(ip[j] + ip[:, None]) + ip[None, :])
        ns = up(up(im[j] + im[:, None]) + im[None, :])
        delta = positive_mul_upper(hp, up(positive_mul_upper(P, ps) + positive_mul_upper(N, ns)))
        error = up(positive_mul_upper(2, delta) / center_abs_lower)
        lower = np.maximum(-1, down(phase_lower - error))
        assert np.all(np.isfinite(lower)) and np.all(lower >= -1) and np.all(lower <= 1)
        # Power-of-two scaling is exact; floor only weakens the lower bound.
        fixed = np.floor(lower * SCALE).astype(np.int64)
        assert L**2 * SCALE < 2**63
        total += int(np.sum(fixed, dtype=np.int64))
    bound = Fraction(total, SCALE * L**3)
    r = Fraction(4, 5)
    A = ((1 - r*r)**-3 + (1 + r*r)**-3) / 2
    assert bound > Fraction(17, 100)
    assert Fraction(17, 100) > 1 / A
    print("phase integral lower =", bound)
    print("phase integral > 17/100; exact 1/A =", 1 / A)
    print("A * phase integral - 1 >", A * Fraction(17, 100) - 1)


if __name__ == "__main__":
    main()
