"""Exact dyadic LDL inertia and residual checks for the Floquet fibre matrices.

The fibre matrices are generated in binary64 and rounded to dyadic rationals;
an exact LDL factorization of the shifted dyadic matrix certifies its inertia,
with an exact residual bound. That the dyadic matrix is close to the exact
fibre matrix is certified separately, in Arb, by regenerate.py.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
from math import isqrt

import numpy as np
from scipy.linalg import ldl
from scipy.special import gamma


P = 8.0
THETA = 0.5
N = 128
J = 12
Q_BITS = 44
Q = 1 << Q_BITS


def psi(x):
    ks = np.arange(1, 60, 2, dtype=float)
    omega = 2 * np.pi / P
    f = -1j / (np.pi * ks)
    c = f * gamma(1j * omega * ks)
    return 2 * np.real(np.exp(1j * omega * np.multiply.outer(np.asarray(x), ks)) @ c)


def image_kernels():
    h = P / N
    s = (np.arange(N) + 0.5) * h
    S, T0 = np.meshgrid(s, s, indexing="ij")
    out = []
    for n in range(-J, J + 1):
        T = T0 + n * P
        d = S - T
        ad = np.abs(d)
        lam = np.minimum(S, T) - np.log1p(np.exp(-ad))
        g = np.exp(-ad / 4) / np.sqrt(1 + np.exp(-ad))
        out.append(g * np.exp(psi(lam)))
    return h, np.asarray(out)


def fiber_matrix(k, h, kernels):
    ns = np.arange(-J, J + 1)
    a = h * np.einsum("n,nij->ij", np.exp(1j * k * ns), kernels, optimize=True)
    return (a + a.conj().T) * 0.5


def rounded_int_array(a):
    scaled = np.rint(np.asarray(a) * Q).astype(np.int64)
    return scaled.astype(object)


def cmul(a, b):
    """Multiply pairs of exact integer matrices (real, imaginary)."""
    ar, ai = a
    br, bi = b
    return ar @ br - ai @ bi, ar @ bi + ai @ br


def cstar(a):
    return a[0].T, -a[1].T


def exact_norm_sq(a, denominator):
    z = sum(int(x) * int(x) for x in a[0].flat)
    z += sum(int(x) * int(x) for x in a[1].flat)
    return Fraction(z, denominator * denominator)


def ceil_sqrt_fraction(x: Fraction, scale: int) -> Fraction:
    # Return ceil(scale*sqrt(x))/scale using integer arithmetic.
    n = x.numerator * scale * scale
    d = x.denominator
    q, r = divmod(n, d)
    root = isqrt(q)
    if root * root * d < n:
        root += 1
    return Fraction(root, scale)


def block_inertia_and_min_singular_sq(dr, di):
    n = dr.shape[0]
    pos = neg = 0
    min_sq = None
    blocks = []
    i = 0
    while i < n:
        if i + 1 < n and (int(dr[i, i + 1]) != 0 or int(di[i, i + 1]) != 0):
            a, c = int(dr[i, i]), int(dr[i + 1, i + 1])
            b_re, b_im = int(dr[i, i + 1]), int(di[i, i + 1])
            det = a * c - b_re * b_re - b_im * b_im
            frob_sq = a * a + c * c + 2 * (b_re * b_re + b_im * b_im)
            if det == 0 or frob_sq == 0:
                raise ArithmeticError(f"singular rounded D block at {i}")
            if det < 0:
                pos += 1
                neg += 1
            elif a + c > 0:
                pos += 2
            elif a + c < 0:
                neg += 2
            else:
                raise ArithmeticError(f"ambiguous trace for D block at {i}")
            sig_sq = Fraction(det * det, Q * Q * frob_sq)
            blocks.append((i, 2, det, frob_sq))
            i += 2
        else:
            d = int(dr[i, i])
            if d == 0:
                raise ArithmeticError(f"zero rounded D pivot at {i}")
            pos += d > 0
            neg += d < 0
            sig_sq = Fraction(d * d, Q * Q)
            blocks.append((i, 1, d, 0))
            i += 1
        min_sq = sig_sq if min_sq is None else min(min_sq, sig_sq)
    # The 1x1/2x2 inertia count is valid only when D has no couplings
    # between the blocks.  Assert this exactly after dyadic rounding.
    paired = {i for i, size, _, _ in blocks if size == 2}
    for i in range(n):
        for j in range(i + 1, n):
            within_pair = j == i + 1 and i in paired
            if not within_pair and (int(dr[i, j]) != 0 or int(di[i, j]) != 0):
                raise ArithmeticError(f"nonzero D entry outside declared blocks: ({i},{j})")
            if not within_pair and (int(dr[j, i]) != 0 or int(di[j, i]) != 0):
                raise ArithmeticError(f"nonzero D entry outside declared blocks: ({j},{i})")
    return pos, neg, min_sq, blocks


def certify_one(a_float, threshold):
    n = a_float.shape[0]
    # Define the exact Hermitian dyadic input B by rounding each entry.
    ar = rounded_int_array(a_float.real)
    ai = rounded_int_array(a_float.imag)
    ai = (ai - ai.T) // 2
    ar = (ar + ar.T) // 2
    for i in range(n):
        ai[i, i] = 0
    t_int = int(round(threshold * Q))
    for i in range(n):
        ar[i, i] -= t_int
    bfloat = np.asarray(ar, dtype=float) / Q + 1j * np.asarray(ai, dtype=float) / Q

    lu, dfloat, perm = ldl(bfloat, lower=True, hermitian=True, overwrite_a=False, check_finite=True)
    lfloat = lu[perm, :]
    # Confirm SciPy's permutation convention against the dyadic target numerically.
    if np.linalg.norm(bfloat[np.ix_(perm, perm)] - lfloat @ dfloat @ lfloat.conj().T, ord="fro") > 1e-7:
        raise ArithmeticError("unexpected scipy.linalg.ldl permutation convention")
    # Work in the same permuted basis as the triangular L factor.
    ar = ar[np.ix_(perm, perm)]
    ai = ai[np.ix_(perm, perm)]

    lr = rounded_int_array(lfloat.real)
    li = rounded_int_array(lfloat.imag)
    dr = rounded_int_array(dfloat.real)
    di = rounded_int_array(dfloat.imag)
    di = (di - di.T) // 2
    dr = (dr + dr.T) // 2
    for i in range(n):
        di[i, i] = 0

    # Exact integer residual: B - L D L* has denominator Q^3.
    ld = cmul((lr, li), (dr, di))
    ldlh = cmul(ld, cstar((lr, li)))
    br = ar * (Q * Q)
    bi = ai * (Q * Q)
    rr = br - ldlh[0]
    ri = bi - ldlh[1]
    rnorm_sq = exact_norm_sq((rr, ri), Q**3)

    # A dyadic C approximating L^{-1}; exact E=I-C L is checked in Z/Q^2.
    cfloat = np.linalg.inv(lfloat)
    cr, ci = rounded_int_array(cfloat.real), rounded_int_array(cfloat.imag)
    cl = cmul((cr, ci), (lr, li))
    er = -cl[0]
    ei = -cl[1]
    for i in range(n):
        er[i, i] += Q * Q
    enorm_sq = exact_norm_sq((er, ei), Q**2)
    if enorm_sq >= 1:
        raise ArithmeticError("inverse residual norm is not below one")
    e_upper = ceil_sqrt_fraction(enorm_sq, 10**15)
    if e_upper >= 1:
        raise ArithmeticError("rounded inverse residual upper bound is not below one")
    cnorm_sq = exact_norm_sq((cr, ci), Q)

    pos, neg, delta_sq, blocks = block_inertia_and_min_singular_sq(dr, di)
    # ||L^-1||_F <= ||C||_F/(1-||I-C L||_2), using Frobenius upper bounds.
    linv_norm_sq_upper = cnorm_sq / ((1 - e_upper) ** 2)
    lhs_sq = linv_norm_sq_upper ** 2 * rnorm_sq
    passed = lhs_sq < delta_sq
    return {
        "pass": passed,
        "positive": int(pos),
        "negative": int(neg),
        "blocks": len(blocks),
        "delta_sq_lower": str(delta_sq),
        "certificate_lhs_sq_upper": str(lhs_sq),
        "factor_residual_frobenius_sq": str(rnorm_sq),
        "inverse_residual_frobenius_sq": str(enorm_sq),
        "inverse_error_upper": str(e_upper),
        "L_inverse_frobenius_sq_upper": str(linv_norm_sq_upper),
        "L_inverse_frobenius_sq_upper_exact": str(linv_norm_sq_upper),
        "factor_residual_frobenius_sq_exact": str(rnorm_sq),
        "D_min_singular_sq_lower_exact": str(delta_sq),
        "delta_lower_float": float(delta_sq) ** 0.5,
        "certificate_lhs_upper_float": float(lhs_sq) ** 0.5,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default="40,111", help="comma-separated mesh indices or 'all'")
    args = ap.parse_args()
    nodes = list(range(129)) if args.nodes == "all" else [int(x) for x in args.nodes.split(",")]
    h, kernels = image_kernels()
    for m in nodes:
        k = m * np.pi / 128
        a = fiber_matrix(k, h, kernels)
        for threshold in (1.0, 3.0):
            result = certify_one(a, threshold)
            print({"node": m, "k": k, "threshold": threshold, **result}, flush=True)


if __name__ == "__main__":
    main()
