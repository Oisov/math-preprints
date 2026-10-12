"""EXACT check of the profile upper bounds in Section 9.6: rho_1(6) <= 0.860918, rho_1(8) <= 0.800639.

For G = sum_k c_k H^k, H = 1/(1-it), the depth remark gives rho_1(p) <= Re B/|B| with
  B_p(G) = (1/pi) int_R (1 - |1-G|^{p-2}(1-G)) conj(G) dt.
Expanding in H and conj(H) reduces B to the integrals
  I(a, b) = int_R (1-it)^{-a} (1+it)^{-b} dt = 2 pi binom(a+b-2, a-1) / 2^{a+b-1}   (a, b >= 1),
and I(a, b) = 0 if a = 0 or b = 0 (a + b >= 2): residue at t = i, or close the contour in the
half-plane without poles.  All arithmetic is in fractions.Fraction (complex numbers as pairs);
a midpoint quadrature in floating point is printed as an independent sanity check.

Standard library only.  Usage: python3 exact_profiles.py
"""
import math
from fractions import Fraction as F

# complex rationals as (re, im)


def cm(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def cadd(a, b):
    return (a[0] + b[0], a[1] + b[1])


# polynomials in X = H, Y = conj(H): dict (i, j) -> complex rational


def padd(p, q, sign=1):
    out = dict(p)
    for k, v in q.items():
        w = out.get(k, (F(0), F(0)))
        out[k] = (w[0] + sign * v[0], w[1] + sign * v[1])
    return out


def pmul(p, q):
    out = {}
    for a, u in p.items():
        for b, v in q.items():
            k = (a[0] + b[0], a[1] + b[1])
            out[k] = cadd(out.get(k, (F(0), F(0))), cm(u, v))
    return out


def B_exact(p, c):
    """B_p(G) for G = sum c[k-1] H^k, exactly; returns (Re B, Im B)."""
    one = {(0, 0): (F(1), F(0))}
    G = {(k + 1, 0): ck for k, ck in enumerate(c)}
    Gb = {(0, k + 1): (ck[0], -ck[1]) for k, ck in enumerate(c)}
    E, Eb = padd(one, G, -1), padd(one, Gb, -1)
    A = one
    for _ in range((p - 2) // 2):
        A = pmul(A, pmul(E, Eb))
    poly = pmul(padd(one, pmul(A, E), -1), Gb)
    re, im = F(0), F(0)
    for (a, b), v in poly.items():
        if v == (0, 0):
            continue
        if a and b:
            # (1/pi) I(a, b) = 2 binom(a+b-2, a-1) / 2^(a+b-1)
            w = F(2 * math.comb(a + b - 2, a - 1), 2 ** (a + b - 1))
            re += v[0] * w
            im += v[1] * w
        else:
            assert a + b >= 2, "constant or linear term would not be integrable"
    return re, im


def B_float(p, c, N=200000):
    cc = [complex(float(x), float(y)) for x, y in c]
    tot = 0j
    for j in range(N):
        th = -math.pi / 2 + (j + 0.5) * math.pi / N
        t = math.tan(th)
        H = 1 / (1 - 1j * t)
        G = sum(ck * H ** (k + 1) for k, ck in enumerate(cc))
        E = 1 - G
        tot += (1 - abs(E) ** (p - 2) * E) * G.conjugate() * (math.pi / N) / math.cos(th) ** 2
    return tot / math.pi


CASES = [
    (6, [(F(11, 8), F(1)), (F(3, 5), F(-2, 5)), (F(-2, 5), F(0))],
     (F(228378383591, 32768000000), F(134956054720, 32768000000)), F(860918, 10**6)),
    (8, [(F(53, 40), F(-7, 8)), (F(13, 20), F(17, 40)), (F(-1, 2), F(-7, 40))],
     (F(24641446872796297211, 3435973836800000000), F(-230501744303958089, 42949672960000000)),
     F(800639, 10**6)),
]


def main():
    ok = True
    for p, c, stated, bound in CASES:
        re, im = B_exact(p, c)
        good = (re, im) == stated and im != 0 and re > 0 and re * re <= bound * bound * (re * re + im * im)
        ok &= good
        bf = B_float(p, c)
        print(f"p = {p}: B = {re} + {im} i  (float check {bf:.10f})")
        print(f"  {'[ok]  ' if good else '[FAIL]'} B matches the paper, Im B != 0, Re B/|B| = "
              f"{float(re) / math.hypot(float(re), float(im)):.9f} <= {bound}")
    print("RESULT:", "ALL OK" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
