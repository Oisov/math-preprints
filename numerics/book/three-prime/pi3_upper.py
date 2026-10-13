"""Certificate for Theorem 9.15: C_3 > 1.599 > 1.5624 > pi*_3.

S_3 = {2^a 3^b 5^c}, c_N = 1/(sqrt(N) log N).  pi*_3 is the norm of
R = (c_{mn}) with rows m in S_3 and columns n in S_3 minus {1}.

Upper bound pi*_3 < 1.5624.  Split R = T + E, T keeping the entries with
mn <= N0 = 10^12 (3429 rows, 3428 columns).
  * ||T|| <= L by Schur's test: (T v)_m <= L u_m and (T^t u)_n <= L v_n for
    the positive weights u, v stored in schur_weights.json (exact binary64
    numbers; see make_weights.py).  Membership mn <= N0 is decided in exact
    integer arithmetic.  The 3429 distinct entries c_N are enclosed in Arb at
    80 bits and rounded up to binary64.  Each of the 6857 sums is then
    computed as math.fsum of binary64 products: every product has relative
    error at most u = 2^-53 and fsum returns the correctly rounded sum of its
    inputs, so the exact sum is at most fsum / (1-u)^2.  This bound and the
    division by the weight are done in Arb.  (All products are positive and
    normal; this is checked.)
  * ||E||^2 <= ||E||_HS^2 <= sum_{N > N0} d(N)/(N log^2 N)
              <= N0^(-1/2) log^(-2) N0 prod_{p=2,3,5} (1 - p^(-1/2))^(-2).

Lower bound C_3 > 1.599.  q is the two-prime witness of Proposition 9.14(3)
(../explicit-bounds/c2_witness.json, checked by its SHA-256 digest), whose norm
||q||_1 <= 1.0005030 is certified by ../explicit-bounds/c2_lower.py.  With
A = ell(q), B = ell(5^(-s) q) enclosed in Arb at 100 bits and t = 1/3,
    C_3 >= (A + t B) / (1.0005030 (1 + t^2/4 + t^4/(64(1 - t^2)))).

Run: python3 pi3_upper.py   (needs python-flint and numpy)
"""
import hashlib
import json
import math
from fractions import Fraction
from pathlib import Path

import numpy as np
from flint import arb, ctx

HERE = Path(__file__).resolve().parent
N0 = 10**12
WITNESS = HERE.parent / "explicit-bounds" / "c2_witness.json"
WITNESS_SHA256 = "a20c5b94cc10ab3a774af6cba3c0d3304aa8acd50f0984a40b5cacec8afe1d77"
Q_NORM = "1.0005030"  # ||q||_1 upper bound of Proposition 9.14(3)


def smooth_numbers(bound):
    return sorted(2**a * 3**b * 5**c for a in range(41) for b in range(26)
                  for c in range(18) if 2**a * 3**b * 5**c <= bound)


def up(x):
    """Binary64 number >= the arb ball x."""
    return math.nextafter(float(x.upper()), math.inf)


def schur_bound(S, u, v):
    ctx.prec = 80
    Sarr = np.array(S, dtype=np.int64)
    cols = Sarr[1:]
    cup = np.array([up(1 / (arb(N).sqrt() * arb(N).log())) if N > 1 else 0.0
                    for N in S])
    fac = 1 / (1 - arb(2) ** -53) ** 2
    tiny = np.finfo(float).tiny
    L = arb(0)
    # rows: (T v)_m = sum over columns n <= N0 // m
    for i, m in enumerate(S):
        k = np.searchsorted(cols, N0 // m, side="right")
        if k == 0:
            continue
        idx = np.searchsorted(Sarr, m * cols[:k])
        assert np.array_equal(Sarr[idx], m * cols[:k])
        p = cup[idx] * v[:k]
        assert np.all(p >= tiny) and np.all(np.isfinite(p))
        L = max(L, (arb(math.fsum(p)) * fac / arb(u[i])).upper())
    # columns: (T^t u)_n = sum over rows m <= N0 // n
    for j, n in enumerate(S[1:]):
        k = np.searchsorted(Sarr, N0 // n, side="right")
        idx = np.searchsorted(Sarr, n * Sarr[:k])
        assert np.array_equal(Sarr[idx], n * Sarr[:k])
        p = cup[idx] * u[:k]
        assert np.all(p >= tiny) and np.all(np.isfinite(p))
        L = max(L, (arb(math.fsum(p)) * fac / arb(v[j])).upper())
    return L


def lower_C3():
    ctx.prec = 100
    raw = WITNESS.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == WITNESS_SHA256
    data = json.loads(raw)
    A = arb(0)
    B = arb(0)
    for (j, k), h in zip(data["index"], data["coefficients_hex"]):
        assert j + k > 0
        a = arb(float.fromhex(h))  # exact dyadic rational
        n = arb(2) ** j * arb(3) ** k
        A += a / (n.sqrt() * n.log())
        B += a / ((5 * n).sqrt() * (5 * n).log())
    t = Fraction(1, 3)
    m = 1 + t**2 / 4 + t**4 / (64 * (1 - t**2))
    assert m == Fraction(1579, 1536)
    return A, B, (A + B / 3) / (arb(Q_NORM) * arb(m.numerator) / m.denominator)


def main():
    A, B, C3 = lower_C3()
    print("ell(q) =", A.str(17), " ell(5^-s q) =", B.str(17))
    print("C_3 >=", C3.lower())
    assert A > arb("1.5487079279756398") and B > arb("0.2877147579857604")
    assert C3 > arb("1.5990217")

    S = smooth_numbers(N0)
    assert len(S) == 3429 and S[0] == 1
    w = json.loads((HERE / "schur_weights.json").read_text())
    assert w["N0"] == N0
    u = np.array([float.fromhex(h) for h in w["u_rows_hex"]])
    v = np.array([float.fromhex(h) for h in w["v_cols_hex"]])
    assert len(u) == len(S) and len(v) == len(S) - 1
    assert np.all(u > 0) and np.all(v > 0)
    L = schur_bound(S, u, v)

    ctx.prec = 80
    zeta2 = arb(1)
    for p in (2, 3, 5):
        zeta2 *= (1 - 1 / arb(p).sqrt()) ** (-2)
    hs = (1 / (arb(N0).sqrt() * arb(N0).log() ** 2) * zeta2).sqrt()
    total = L + hs
    print("N0 =", N0, " |S_3 cap [1,N0]| =", len(S))
    print("binary64 largest singular value of T:", float.fromhex(w["float_singular_value"]))
    print("certified ||T|| <=", L)
    print("certified ||E|| <=", hs.upper())
    print("certified pi*_3 <=", total.upper())
    print("C_3/pi*_3 >", (arb("1.5990217") / total).lower())
    assert L < arb("1.56179569435114")
    assert hs < arb("0.000529")
    assert total < arb("1.56232458") and total < arb("1.5624")
    assert arb("1.5990217") / total > arb("1.02348")
    print("OK: C_3 > 1.5990217 > 1.599 > 1.5624 > pi*_3")


if __name__ == "__main__":
    main()
