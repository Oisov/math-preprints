"""The lower bound |a| > 2 sqrt(p-1)/p: Lemma 11, identity (10) and Theorem 3.

Run: python3 check_bound.py
"""

import numpy as np
from numpy.polynomial import polynomial as P

from opa import Profile, check, circle_rule, f_alpha, heading, opa, table, zeros


def cone_ratio(p, n_t=2001, n_theta=2000):
    """max |Im w| / (k |w|) and min Re w over a grid of E = t e^{i theta}."""
    k = abs(p - 2) / p
    t = np.geomspace(1e-4, 1e4, n_t)[:, None]
    th = np.linspace(0, np.pi, n_theta + 1)[1:][None, :]
    E = t * np.exp(1j * th)
    E = E[np.abs(1 - E) > 1e-9]
    K = np.abs(E) ** (p - 2) * E
    w = (1 - K) / (1 - E)
    return np.max(np.abs(w.imag) / (k * np.abs(w))), np.min(w.real / np.abs(w))


def main():
    ok = True

    heading("The cone lemma: |arg w| <= arcsin(|p-2|/p)", "Lemma 11")
    rows, good = [], True
    for p in [1.05, 1.2, 1.5, 1.9, 2.1, 3, 4, 8, 30]:
        ratio, re_min = cone_ratio(p)
        rows.append([p, f"{np.degrees(np.arcsin(abs(p - 2) / p)):.3f}", f"{ratio:.6f}",
                     f"{re_min:+.4f}"])
        good &= ratio <= 1 + 1e-9 and re_min > 0
    table(["p", "arcsin(k) in degrees", "max |Im w|/(k|w|)", "min Re w/|w|"], rows)
    print("  (grid of 4 million points E = t e^{i theta}, 1e-4 <= t <= 1e4)")
    ok &= check("ratio <= 1, approached from below; Re w > 0", good)

    heading("Identity (10) on actual OPAs", "proof of Theorem 3")
    rows, good = [], True
    examples = [(4.0, Profile([4, -6, 3]), 0.9999), (1.5, Profile.family(10, 1.5), 0.999),
                (1.2, Profile([2, -1]), 0.999), (3.0, Profile.family(10, 3), 0.99)]
    for p, ph, a in examples:
        z, wts = circle_rule(1 - a)
        f = f_alpha(ph, a)(z)
        c, _ = opa(f, z, wts, 1, p)
        root = zeros(c)[0]
        fz = c[1] * f  # normalize the OPA to z - root
        E = 1 - (z - root) * fz
        K = np.abs(E) ** (p - 2) * E
        w = (1 - K) / (1 - E)
        val = wts @ (w * np.abs(fz) ** 2 * (1 - root * np.conj(z)))
        scale = wts @ np.abs(w * np.abs(fz) ** 2)
        rows.append([p, a, f"{abs(root):.8f}", f"{abs(val) / scale:.1e}"])
        good &= abs(val) / scale < 1e-8
    table(["p", "alpha", "|a|", "|int w|f|^2(1-a conj z)| / int |w||f|^2"], rows)
    ok &= check("the integral vanishes", good)

    heading("No OPA zero below the bound: random targets", "Theorem 3")
    rng = np.random.default_rng(2026)
    z, wts = circle_rule(uniform=512)
    rows, good = [], True
    for p in [1.2, 1.5, 3.0, 4.0, 6.0]:
        smallest, inside, total = np.inf, 0, 0
        for _ in range(150):
            deg = rng.integers(1, 9)
            g = rng.normal(size=deg + 1) + 1j * rng.normal(size=deg + 1)
            for n in [1, 2, 3]:
                rts = zeros(opa(P.polyval(z, g), z, wts, n, p)[0])
                if len(rts):
                    smallest = min(smallest, np.min(np.abs(rts)))
                    inside += np.sum(np.abs(rts) < 1)
                    total += len(rts)
        bound = 2 * np.sqrt(p - 1) / p
        rows.append([p, f"{bound:.6f}", f"{smallest:.6f}", f"{inside}/{total}"])
        good &= smallest > bound
    table(["p", "2 sqrt(p-1)/p", "smallest |zero| found", "zeros in D"], rows)
    print("  (150 random complex polynomial targets of degree <= 8, OPAs of degree 1-3)")
    ok &= check("every zero lies outside the disc of radius 2 sqrt(p-1)/p", good)
    return ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
