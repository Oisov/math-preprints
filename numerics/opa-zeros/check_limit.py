"""Checks of the limit problem: Lemmas 1-7, Corollary 2, Sections 7.1 and 7.3.

Run: python3 check_limit.py
"""

import numpy as np
from scipy.special import beta

from opa import (F_sigma_closed_form, F_sigma_quadrature, Profile, check, heading,
                 lambda_threshold, scaled_criterion, table)


def main():
    ok = True

    heading("The beta integral F_sigma", "Lemma 6")
    rows = []
    for p in [1.1, 1.5, 1.9, 2.1, 3, 4, 6]:
        a, b = F_sigma_closed_form(p), F_sigma_quadrature(p)
        rows.append([p, f"{a:.12f}", f"{b:.12f}", f"{abs(a - b):.1e}"])
        ok &= abs(a - b) < 1e-9 and a > 0
    table(["p", "closed form", "quadrature", "difference"], rows)
    ok &= check("F_sigma > 0 and closed form = quadrature", ok)

    heading("Boundary identity: (1) at v = 1/2 equals (5)", "Lemma 3")
    rows, good = [], True
    profiles = [Profile([4, -6, 3], "4-6w+3w^2"), Profile([2, -1], "2-w"),
                Profile([3.88, -8.785, 5.905], "3.88-8.785w+5.905w^2"),
                Profile([0.5, 0.2, 0.3], "0.5+0.2w+0.3w^2")]
    for ph in profiles:
        for p in [1.3, 1.7, 2.6, 4.0]:
            a, b = ph.dpsi(0.5, p), ph.criterion(p)
            rows.append([ph.name, p, f"{a:+.10f}", f"{b:+.10f}"])
            good &= abs(a - b) < 1e-8
    table(["phi", "p", "d/dx Psi(1/2) by (1)", "by (5)"], rows)
    ok &= check("both formulas agree", good)

    heading("p = 2: the derivative at the line Re v = 1/2 is 1", "Corollary 2")
    vals = [ph.criterion(2.0) for ph in profiles]
    print("  " + ", ".join(f"{v:.12f}" for v in vals))
    ok &= check("Psi'(1/2) = 1 for every profile", np.allclose(vals, 1, atol=1e-10))

    heading("p = 4: Psi'(1/2) = 1 - lambda + 3 lambda^2 - lambda^3", "Section 7.1")
    rows, good = [], True
    for lam in [0.5, 1, 2, 2.769, 3, 5, 10]:
        q = Profile.family(lam, 4).criterion(4)
        exact = 1 - lam + 3 * lam ** 2 - lam ** 3
        rows.append([lam, f"{q:+.10f}", f"{exact:+.10f}"])
        good &= abs(q - exact) < 1e-8 * max(1, abs(exact))
    table(["lambda", "quadrature", "cubic"], rows)
    ok &= check("quadrature matches the cubic", good)
    root = np.roots([-1, 3, -1, 1])
    lam0 = root[np.abs(root.imag) < 1e-12].real[0]
    print(f"  real zero of the cubic: lambda_0 = {lam0:.10f}")
    ph3 = Profile([4, -6, 3])
    ok &= check(f"phi_3 = 4-6w+3w^2: Psi'(1/2) = {ph3.criterion(4):.12f} = -2",
                abs(ph3.criterion(4) + 2) < 1e-10)
    ok &= check(f"phi_3 has zeros of modulus sqrt(4/3) = {np.sqrt(4/3):.6f}: "
                f"{ph3.min_modulus():.6f}", abs(ph3.min_modulus() - np.sqrt(4 / 3)) < 1e-12)

    heading("1 < p < 4/3: phi = 2 - w", "Section 7.1")
    def M(a):  # int_T |1-w|^a dm = 2^a I(a, 0)
        return 2 ** a * beta((a + 1) / 2, 0.5) / np.pi

    rows, good = [], True
    for p in [1.1, 1.2, 1.3, 4 / 3, 1.4, 1.6]:
        formula = p * 2 ** (1 - p) * (M(p) - M(p - 2))
        q = Profile([2, -1]).criterion(p)
        rows.append([f"{p:.4f}", f"{q:+.10f}", f"{formula:+.10f}"])
        good &= abs(q - formula) < 1e-8
    table(["p", "quadrature", "p 2^(1-p) (M(p)-M(p-2))"], rows)
    ok &= check("formula correct; negative exactly for p < 4/3", good)

    heading("Thresholds lambda_0(p) quoted in Section 7.3", "Section 7.3")
    quoted = {1.8: 7.3, 1.95: 37, 1.99: 197, 2.01: 202, 2.05: 42, 2.2: 12, 3: 3.9, 4: 2.77}
    rows, good = [], True
    for p, q in quoted.items():
        found = lambda_threshold(p)
        lam = found[0] if len(found) == 1 else float("nan")
        rows.append([p, f"{lam:.4f}", q, f"{2 / abs(p - 2):.1f}", len(found)])
        good &= len(found) == 1 and abs(lam - q) / q < 0.02
    table(["p", "lambda_0 (computed)", "quoted", "2/|p-2|", "sign changes"], rows)
    ok &= check("one sign change, matching the quoted values", good)

    heading("phi_lambda has no zeros in the closed disc", "Lemma 5")
    rows, good = [], True
    for p in [1.5, 3]:
        for lam in [0.5, 3, 30, 300]:
            m = Profile.family(lam, p).min_modulus()
            predicted = np.sqrt(1 + 1 / lam)
            rows.append([p, lam, f"{m:.6f}", f"{predicted:.6f}"])
            good &= m > 1 and abs(m - predicted) < 1e-9
    table(["p", "lambda", "min |zero|", "sqrt(1+1/lambda)"], rows)
    ok &= check("all zeros outside the closed disc", good)

    heading("The scaled criterion (6) tends to -F_sigma", "Lemma 7")
    rows = []
    for p in [1.5, 3]:
        for lam in [10, 100, 1000, 10000]:
            rows.append([p, lam, f"{scaled_criterion(1 / lam, p):+.8f}",
                         f"{-F_sigma_closed_form(p):+.8f}"])
    table(["p", "lambda", "(6)", "-F_sigma"], rows)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
