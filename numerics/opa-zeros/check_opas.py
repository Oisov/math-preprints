"""OPAs of actual targets: Lemma 2, Theorem 1, Corollary 1 and Section 7.5.

Run: python3 check_opas.py
"""

import numpy as np
from numpy.polynomial import polynomial as P

from opa import Profile, check, circle_rule, f_alpha, heading, opa, table, zeros

G16 = [10, -15, 22, -29, 36, -41, 44, -45, 44, -41, 36, -30, 23, -17, 12, -7, 4]


def degree_one_zero(f, delta, p, fine=False):
    z, w = circle_rule(delta, panel_nodes=40, ratio=1.3) if fine else circle_rule(delta)
    c, err = opa(f(z), z, w, 1, p)
    return zeros(c)[0], err


def main():
    ok = True

    heading("Depth of the zero of the degree-one OPA to 1/f_alpha", "Lemma 2")
    print("  Entries are (1-|zero|)/(1-alpha); positive means the zero lies in D.")
    print("  By the proof of Lemma 2 they tend to 2 v_* - 1 as alpha -> 1, but slowly:")
    print("  for p = 1.2 the correction decays roughly like (1-alpha)^0.2. The column")
    print("  'Aitken' extrapolates the last three entries.\n")
    cases = [(4.0, Profile([4, -6, 3], "4-6w+3w^2")),
             (4.0, Profile.family(10, 4)), (3.0, Profile.family(10, 3)),
             (1.8, Profile.family(30, 1.8)), (1.5, Profile.family(10, 1.5)),
             (1.2, Profile([2, -1], "2-w"))]
    alphas = [0.9, 0.99, 0.999, 0.9999, 0.99999]
    rows, good, worst = [], True, 0.0
    for p, ph in cases:
        vs = ph.vstar(p)
        depths = []
        for a in alphas:
            r, _ = degree_one_zero(f_alpha(ph, a), 1 - a, p)
            r2, _ = degree_one_zero(f_alpha(ph, a), 1 - a, p, fine=True)
            worst = max(worst, abs(r - r2) / (1 - a))
            depths.append((1 - abs(r)) / (1 - a))
        x0, x1, x2 = depths[-3:]
        aitken = x2 - (x2 - x1) ** 2 / ((x2 - x1) - (x1 - x0))
        rows.append([p, ph.name] + [f"{d:+.4f}" for d in depths]
                    + [f"{aitken:+.4f}", f"{2 * vs - 1:+.4f}"])
        good &= vs > 0.5 and min(depths[2:]) > 0
        good &= abs(aitken - (2 * vs - 1)) < 0.2 * (2 * vs - 1)
    table(["p", "profile"] + [f"alpha={a}" for a in alphas] + ["Aitken", "2v*-1"], rows)
    print(f"\n  two quadrature rules agree to {worst:.1e} (relative to 1-alpha)")
    ok &= check("zeros in D for alpha >= 0.999; extrapolated depth near 2v*-1", good)

    heading("Theorem 1 with an honest polynomial target", "proof of Theorem 1")
    p, ph, a = 1.2, Profile([2, -1], "2-w"), 0.99
    f = f_alpha(ph, a)
    M = 2 ** 15
    zz = np.exp(2j * np.pi * np.arange(M) / M)
    taylor = np.fft.fft(f(zz)) / M  # Taylor coefficients of f_alpha (aliasing ~ alpha^M)
    z, w = circle_rule(1 - a)
    rows, good = [], True
    min_f = np.min(np.abs(f(zz)))
    for N in [250, 500, 1000, 2000, 4000]:
        gN = taylor[: N + 1].real
        sup_err = np.max(np.abs(P.polyval(zz, gN) - f(zz)))
        r = zeros(opa(P.polyval(z, gN), z, w, 1, p)[0])[0]
        zero_free = sup_err < min_f
        rows.append([N, f"{sup_err:.2e}", "yes" if zero_free else "not shown", f"{abs(r):.10f}"])
    table(["Taylor degree N", "sup |g_N - f_alpha|", "g_N zero-free", "|zero of OPA|"], rows)
    print(f"  (p = 1.2, phi = 2-w, alpha = 0.99, min |f_alpha| on T = {min_f:.4f})")
    ok &= check("for large N, g_N is zero-free and its OPA vanishes in D",
                rows[-1][2] == "yes" and float(rows[-1][3]) < 1)

    heading("Degree n: the OPA to 1/g(z^n) is A + B z^n", "Corollary 1")
    z, w = circle_rule(uniform=8192)
    c1, _ = opa(P.polyval(z, G16), z, w, 1, 4)
    rows, good = [], True
    for n in [2, 3, 5]:
        cn, _ = opa(P.polyval(z ** n, G16), z, w, n, 4)
        expected = np.zeros(n + 1, complex)
        expected[0], expected[n] = c1
        dev = np.max(np.abs(cn - expected))
        rad = np.abs(zeros(cn))
        rows.append([n, f"{dev:.1e}", f"{rad.min():.8f}", f"{rad.max():.8f}",
                     f"{abs(zeros(c1)[0]) ** (1 / n):.8f}"])
        good &= dev < 1e-10 and rad.max() < 1
    table(["n", "max |c - (A,0,..,0,B)|", "min |zero|", "max |zero|", "|a|^(1/n)"], rows)
    ok &= check("n zeros in D on the circle of radius |a|^(1/n)", good)

    heading("The certified example of Section 7.2, numerically", "Section 7.2")
    c, err = opa(P.polyval(z, G16), z, w, 1, 4)
    r = zeros(c)[0]
    print(f"  OPA = {c[0].real:.10f} + {c[1].real:.10f} z,  zero = {r.real:.8f},"
          f"  error^4 = {err ** 4:.10f}")
    print(f"  smallest |zero of g| = {np.min(np.abs(P.polyroots(G16))):.6f},"
          f"  min_T |g| = {np.min(np.abs(P.polyval(z, G16))):.6f}")
    ok &= check("zero inside D", abs(r) < 1)

    heading("The deeper non-outer example of Section 7.5", "Section 7.5")
    ph = Profile([3.88, -8.785, 5.905], "3.88-8.785w+5.905w^2")
    r, _ = degree_one_zero(f_alpha(ph, 0.91), 0.09, 4)
    print(f"  phi = {ph.name}, alpha = 0.91, p = 4: |zero| = {abs(r):.6f}")
    print(f"  phi has a zero of modulus {ph.min_modulus():.4f} < 1, so f_alpha is not outer")
    print(f"  lower bound of Theorem 3: 2 sqrt(p-1)/p = {np.sqrt(3) / 2:.6f}")
    ok &= check("|zero| = 0.9968 as quoted", abs(abs(r) - 0.9968) < 5e-5)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
