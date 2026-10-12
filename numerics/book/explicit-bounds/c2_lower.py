"""Prop 9.14(3): rigorous lower bound C_2 >= ell(q)/||q||_1 for the stored witness q.

q(z_2, z_3) = sum_{(j,k)} a_{jk} z_2^j z_3^k with the binary64 coefficients in
c2_witness.json (each binary64 number is an exact dyadic rational).

* ell(q) = sum a_{jk} / (sqrt(n) log n), n = 2^j 3^k, is enclosed in Arb.
* ||q||_1 is bounded by the midpoint rule on an M x M grid of the torus with
  explicit first- and second-order Taylor errors:
      ||q||_1 <= mean|q(mid)| + (h/4) mean(|q_theta| + |q_phi|) + R2,
  R2 = (1/2)(h^2/12 sum j^2|a| + 2 h^2/16 sum jk|a| + h^2/12 sum k^2|a|).
* The grid values are computed in binary64 as V1 C V2^T, where the matrices of
  exponentials V are enclosed in Arb and rounded. The total floating-point
  error is bounded a priori, using only the IEEE-754 model for +, *
  (Higham, Accuracy and Stability, Thm 3.5 and Sec. 4): no allowance factor.

Requires numpy and python-flint.  Usage: python3 c2_lower.py [M]  (default 8192)
"""
import json
import sys
from pathlib import Path

import numpy as np
from flint import arb, acb, ctx

ctx.prec = 80
U = 2.0**-53


def gamma(n):
    return n * U / (1 - n * U)


def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 8192
    data = json.loads(Path(__file__).with_name("c2_witness.json").read_text())
    idx = [tuple(x) for x in data["index"]]
    a = np.array([float.fromhex(h) for h in data["coefficients_hex"]])
    D = max(max(j, k) for j, k in idx)
    C = np.zeros((D + 1, D + 1))
    for (j, k), v in zip(idx, a):
        C[j, k] = v
    S1 = float(np.sum(np.abs(C)))

    # exponentials e^{i j theta_m}, theta_m = 2 pi (m + 1/2)/M, enclosed in Arb
    rad = 0.0
    V = np.zeros((M, D + 1), complex)
    two_pi = 2 * arb.pi()
    for m in range(M):
        th = two_pi * (m + arb(1) / 2) / M
        for j in range(D + 1):
            z = acb(0, j * th).exp()
            re, im = float(z.real.mid()), float(z.imag.mid())
            V[m, j] = complex(re, im)
            err = (abs(z.real - re) + abs(z.imag - im)).upper()
            rad = max(rad, float(err))
    # |V_float - V_exact| <= rad entrywise, |V_exact| = 1
    J = np.arange(D + 1)

    def sum_abs(V1, Cm, V2):
        """mean over the grid of |V1 Cm V2^T|, computed in row blocks."""
        tot = 0.0
        for s in range(0, M, 512):
            B = V1[s:s + 512] @ Cm @ V2.T
            tot += float(np.sum(np.abs(B)))
        return tot / (M * M)

    meanP = sum_abs(V, C.astype(complex), V)
    meanPt = sum_abs(V * (1j * J), C.astype(complex), V)
    meanPp = sum_abs(V, C.astype(complex), V * (1j * J))

    # a priori error per grid value of P, P_theta, P_phi (complex arithmetic):
    # exact inputs perturbed by rad: |dV1 C V2| + ... <= (2 rad + rad^2) sum|c| (times D for derivatives);
    # matrix-product rounding: complex products cost gamma(2(D+1)+2) |V1||C||V2| (Higham, complex case).
    n_ops = 2 * (D + 1) + 4
    eP = ((2 * rad + rad**2) + gamma(n_ops) * (1 + rad) ** 2) * S1
    radD = rad + 2 * U          # V * (i j) is formed in binary64: one more rounding
    eD = D * ((2 * radD + radD**2) + gamma(n_ops) * (1 + radD) ** 2) * S1
    # summation of M^2 nonnegative terms and |.| in binary64
    summ = gamma(M * M + 4)
    h = 2 * np.pi / M
    jj = np.array([j for j, k in idx]); kk = np.array([k for j, k in idx]); aa = np.abs(a)
    R2 = 0.5 * (np.sum(jj**2 * aa) * h * h / 12 + 2 * np.sum(jj * kk * aa) * h * h / 16
                + np.sum(kk**2 * aa) * h * h / 12)
    first = meanP * (1 + summ) + eP
    grad = ((meanPt + meanPp) * (1 + summ) + 2 * eD) * h / 4
    L1up = arb(first) + arb(grad) + arb(R2) * (1 + arb(10) ** -12)
    L1up = arb(float(L1up.upper()) * (1 + 4 * U))

    ell = arb(0)
    for (j, k), v in zip(idx, a):
        n = arb(2) ** j * arb(3) ** k
        ell += arb(v) / (n.sqrt() * n.log())
    bound = ell / L1up
    print(f"M = {M}, exponential enclosure radius {rad:.2e}, float error per value {eP:.2e}")
    print("ell(q) in", ell)
    print("||q||_1 <=", float(L1up.upper()), f"(grid {meanP:.10f}, gradient {grad:.3e}, R2 {R2:.3e})")
    print("C_2 >=", float(bound.lower()))
    return float(bound.lower())


if __name__ == "__main__":
    main()
