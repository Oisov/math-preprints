"""Regenerate the Remark 5.6 certificate from scratch (stored in floquet_gap.json).

For each Bloch node m = 0, ..., 128 (k = m pi/128) this
* encloses every entry of the 128 x 128 fibre matrix in Arb (images |n| <= 12),
* bounds the distance from the dyadic matrix used in dyadic_ldl.py, and
* certifies, at the thresholds 1 and 3, the inertia of the shifted fibre matrix
  by an exact LDL factorization plus a residual inequality that absorbs both
  the LDL residual and the entry error.
floquet_gap.py then checks the stored rows and the operator error budget
(mesh, midpoint and image-cutoff errors) in exact rational arithmetic.

Needs python-flint, numpy and scipy. About 40 s per node; --nodes all runs all
129 nodes (about 10 minutes when split over 16 processes, see README).
Usage: python3 regenerate.py --nodes all   (or --nodes 0,17,40)
"""
from __future__ import annotations

import argparse
import importlib.util
from fractions import Fraction
from pathlib import Path

import numpy as np
from flint import acb, acb_mat, arb, ctx


HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("floquet_dyadic", HERE / "dyadic_ldl.py")
dyadic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dyadic)

N, P, J, Q = 128, 8, 12, 1 << dyadic.Q_BITS


def exact_kernel_images():
    ctx.prec = 128
    pi = arb.pi()
    omega = pi / 4
    coeffs = []
    for k in range(1, 20, 2):
        f = acb(0, -1) / (pi * k)
        coeffs.append((k, f * acb(0, omega * k).gamma()))
    # For odd k >= 21, a_k <= (4 sqrt(3)/pi) k^(-3/2) q^k,
    # q=e^(-pi^2/8)<0.3; the geometric sum is < 3e-13.
    psi_tail = arb(0, "1e-12")

    def psi(lam):
        z = acb(0)
        for k, c in coeffs:
            z += c * acb(0, omega * k * lam).exp()
        return 2 * z.real + psi_tail

    h = arb((1, -4))
    images = []
    for n in range(-J, J + 1):
        M = acb_mat(N, N)
        for i in range(N):
            snum = 2 * i + 1
            s = arb((snum, -5))
            for j in range(N):
                tnum = 2 * j + 1 + 256 * n
                t = arb((tnum, -5))
                dnum = snum - tnum
                ad = arb((abs(dnum), -5))
                emd = (-ad).exp()
                low = min(snum, tnum)
                lam = arb((low, -5)) - (1 + emd).log()
                g = (-ad / 4).exp() / (1 + emd).sqrt()
                kval = g * psi(lam).exp()
                M[i, j] = acb(kval, 0)
        images.append(M)
        print(f"finished Arb image n={n}", flush=True)
    return h, pi, images


def fiber_ball(m, h, pi, images):
    A = acb_mat(N, N)
    for idx, n in enumerate(range(-J, J + 1)):
        phase = acb(0, (pi * (m * n)) / 128).exp()
        A += images[idx] * phase
    A = A * h
    # Both mirrored balls enclose the same exact Hermitian entry; their
    # arithmetic mean still encloses it. Reflect that averaged upper half.
    for i in range(N):
        A[i, i] = acb(A[i, i].real, 0)
        for j in range(i + 1, N):
            v = (A[i, j] + A[j, i].conjugate()) / 2
            A[i, j] = v
            A[j, i] = v.conjugate()
    return A


def dyadic_target(a):
    ar = np.rint(a.real * Q).astype(np.int64).astype(object)
    ai = np.rint(a.imag * Q).astype(np.int64).astype(object)
    ar = (ar + ar.T) // 2
    ai = (ai - ai.T) // 2
    for i in range(N):
        ai[i, i] = 0
    return ar, ai


def max_entry_error(A, ar, ai):
    err = Fraction(0)
    for i in range(N):
        for j in range(N):
            b = acb(arb((int(ar[i, j]), -dyadic.Q_BITS)), arb((int(ai[i, j]), -dyadic.Q_BITS)))
            q = (A[i, j] - b).abs_upper().upper().fmpq()
            e = Fraction(int(q.numerator), int(q.denominator))
            if e > err:
                err = e
    return err


def combined_certificate(cert, entry_error):
    linv_sq = Fraction(cert["L_inverse_frobenius_sq_upper_exact"])
    res_sq = Fraction(cert["factor_residual_frobenius_sq_exact"])
    delta_sq = Fraction(cert["D_min_singular_sq_lower_exact"])
    res_up = dyadic.ceil_sqrt_fraction(res_sq, 10**24)
    err_frob_up = N * entry_error
    alpha = linv_sq * (res_up + err_frob_up)
    return alpha * alpha < delta_sq, alpha, delta_sq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default="40,111", help="comma-separated Bloch mesh indices or all")
    ap.add_argument("--compare", action="store_true",
                    help="check each regenerated row against floquet_gap.json")
    args = ap.parse_args()
    stored = {}
    if args.compare:
        import json
        stored = {r["node"]: r for r in json.loads((HERE / "floquet_gap.json").read_text())["nodes"]}
    nodes = list(range(129)) if args.nodes == "all" else [int(x) for x in args.nodes.split(",")]
    h, pi, images = exact_kernel_images()
    fh, fk = dyadic.image_kernels()
    results = []
    for m in nodes:
        kf = m * np.pi / 128
        afloat = dyadic.fiber_matrix(kf, fh, fk)
        ar, ai = dyadic_target(afloat)
        A = fiber_ball(m, h, pi, images)
        error = max_entry_error(A, ar, ai)
        row = {"node": m, "max_entry_error_upper": str(error)}
        for t in (1.0, 3.0):
            cert = dyadic.certify_one(afloat, t)
            passed, alpha, delta_sq = combined_certificate(cert, error)
            row[f"threshold_{int(t)}"] = {
                "inertia": [cert["positive"], cert["negative"]],
                "combined_residual_pass": passed,
                "Linv2_times_total_residual_upper": str(alpha),
                "D_min_singular_sq_lower": str(delta_sq),
            }
        results.append(row)
        if args.compare and row != stored[m]:
            raise SystemExit(f"node {m}: regenerated row differs from floquet_gap.json")
        print(row, flush=True)
    print({"nodes": len(results), "all_combined_certificates_pass": all(
        r[f"threshold_{t}"]["combined_residual_pass"] for r in results for t in (1, 3)
    ), **({"matches_floquet_gap_json": True} if args.compare else {})}, flush=True)


if __name__ == "__main__":
    main()
