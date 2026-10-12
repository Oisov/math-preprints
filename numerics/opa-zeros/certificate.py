"""Exact certificate for the example of Section 7.2 (p = 4).

Claim: the OPA to 1/g of degree one in H^4 vanishes in D, where g is the
integer polynomial G16 below, and g has no zeros in the closed disc.

Everything is done in exact rational arithmetic with the standard library;
no floating point enters any inequality. Floats are printed for reading only.

Run: python3 certificate.py
"""

from fractions import Fraction as Fr

G16 = [10, -15, 22, -29, 36, -41, 44, -45, 44, -41, 36, -30, 23, -17, 12, -7, 4]
G14 = [10, -15, 21, -28, 34, -38, 41, -41, 39, -35, 30, -23, 17, -10, 5]


def mul(a, b):
    c = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        for j, y in enumerate(b):
            c[i + j] += x * y
    return c


def schur_cohn_zero_free(f):
    """True iff the polynomial f (ascending coefficients) has no zeros in |z| <= 1.

    If |f_0| > |f_n|, then h = f_0 f - f_n f^*, with f^*(z) = z^n f(1/z), has
    degree n-1 and, by Rouche's theorem, the same zeros as f in the closed
    disc. Iterating down to a nonzero constant proves that f has none.
    """
    f = [Fr(x) for x in f]
    steps = 0
    while len(f) > 1:
        if abs(f[0]) <= abs(f[-1]):
            return False, steps
        h = [f[0] * a - f[-1] * b for a, b in zip(f, f[::-1])]
        assert h[-1] == 0
        f = h[:-1]
        steps += 1
    return f[0] != 0, steps


def Phi(g, A, B):
    """||1 - (A + Bz) g||_{H^4}^4 = sum of squared coefficients of (1 - (A+Bz) g)^2."""
    F = [-x for x in mul([A, B], g)]
    F[0] += 1
    return sum(x * x for x in mul(F, F))


def line_quartic(g, s):
    """Coefficients of t -> Phi(t, s t) (ascending), computed exactly.

    With H = (1 + s z) g and S = H^2, the polynomial 1 - tH squared has
    coefficients [k=0] - 2t H_k + t^2 S_k, and summing their squares gives
    the quartic below.
    """
    H = mul([1, s], g)
    S = mul(H, H)
    H = H + [0] * (len(S) - len(H))
    return [1,
            -4 * H[0],
            4 * sum(h * h for h in H) + 2 * S[0],
            -4 * sum(h * x for h, x in zip(H, S)),
            sum(x * x for x in S)]


def tangent_lower_bound(q, t1, t2):
    """For a convex quartic q with q'(t1) < 0 < q'(t2), return L with q >= L on R.

    On [t1, t2] the tangent at t1 gives q(t) >= q(t1) + q'(t1)(t - t1)
    >= q(t1) + q'(t1)(t2 - t1); to the left of t1 the function decreases and to
    the right of t2 it increases.
    """
    def val(t):
        return sum(c * t ** k for k, c in enumerate(q))

    def der(t):
        return sum(k * c * t ** (k - 1) for k, c in enumerate(q) if k)

    assert der(t1) < 0 < der(t2), "derivative does not change sign on [t1, t2]"
    return val(t1) + der(t1) * (t2 - t1)


def certify(name, g, P0, plus, minus):
    print(f"\n{name}: g = " + " ".join(f"{c:+d}z^{k}" for k, c in enumerate(g)))
    ok, steps = schur_cohn_zero_free(g)
    print(f"  1. Schur-Cohn: {steps} reduction steps end in a nonzero constant: {ok}")
    A0, B0 = P0
    phi0 = Phi(g, A0, B0)
    print(f"  2. P0 = ({A0}, {B0}),  |A0| < |B0|: {abs(A0) < abs(B0)}")
    print(f"     Phi(P0) = {phi0}  ~ {float(phi0):.10f}")
    bounds = []
    for s, (t1, t2) in [(1, plus), (-1, minus)]:
        q = line_quartic(g, s)
        assert all(Phi(g, Fr(t), s * Fr(t)) == sum(c * Fr(t) ** k for k, c in enumerate(q))
                   for t in range(-2, 3)), "quartic formula mismatch"
        L = tangent_lower_bound(q, t1, t2)
        bounds.append(L)
        sign = "+" if s == 1 else "-"
        print(f"  3. Phi(t, {sign}t) = " + " ".join(f"{c:+d}t^{k}" for k, c in enumerate(q)))
        print(f"     on [{t1}, {t2}] the tangent bound gives Phi >= {float(L):.10f}")
    margin = min(bounds) - phi0
    verdict = ok and abs(A0) < abs(B0) and margin > 0
    print(f"  4. min over both lines - Phi(P0) = {float(margin):.3e} > 0: {margin > 0}")
    print(f"  => the OPA A + Bz has |A| < |B|; its zero -A/B lies in D: {verdict}")
    return verdict


def main():
    print(__doc__.split("\n\n")[1])
    print("\nWhy steps 2-4 suffice. Since g is real, the OPA is real, and it minimizes the")
    print("convex function Phi on R^2. The OPA has no zero in D iff |A| >= |B|, a union")
    print("of two closed convex cones bounded by the lines A = B and A = -B. If the")
    print("minimizer lay in one of them, the segment from it to P0 would cross a line at")
    print("a point Y with Phi(Y) <= Phi(P0) by convexity; steps 3-4 rule this out.")
    ok = certify("Degree 16 (Section 7.2)", G16, (Fr(79, 5000), Fr(1583, 100000)),
                 (Fr(157, 10000), Fr(79, 5000)), (Fr(0), Fr(1, 1000)))
    ok &= certify("Degree 14 (mentioned in Section 7.2)", G14,
                  (Fr(8109, 500000), Fr(16223, 1000000)),
                  (Fr(1621, 100000), Fr(811, 50000)), (Fr(0), Fr(1, 1000)))
    print("\nAll certificates verified." if ok else "\nA certificate FAILED.")
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
