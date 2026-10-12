"""Combine rigor.json and samples_*.json into the final probabilistic certificate.

Usage: python certify.py samples_SEED_NB.json [log10(1/delta) per stage, default 15]
Every quantity below is an upper bound computed in exact rational arithmetic
(Fractions), except square roots/logs which are rounded upward explicitly.
"""
import sys, json, math, os
from fractions import Fraction as Fr

HERE = os.path.dirname(os.path.abspath(__file__))
u = Fr(1, 2 ** 53)


def gam(n):
    return n * u / (1 - n * u)


def sqrt_up(x):
    x = Fr(x)
    sc = 1 << 200
    return Fr(math.isqrt(-((-x.numerator * sc * sc) // x.denominator)) + 1, sc) if x > 0 else Fr(0)


def ln_up(y):      # upper bound for ln(y), y >= 1 integer power of 10 style input
    return Fr(math.log(y)) * (1 + Fr(1, 10 ** 12))


def main():
    R = json.load(open(os.path.join(HERE, 'rigor.json')))
    S = json.load(open(sys.argv[1] if os.path.isabs(sys.argv[1]) else os.path.join(HERE, sys.argv[1])))
    kd = int(sys.argv[2]) if len(sys.argv) > 2 else 15
    m, k = R['m'], R['k']
    M = Fr(R['M_upper']); alpha = Fr(R['alpha_upper'])
    num_lo = Fr(R['numerator'][0]); pi_hi = Fr(R['pi'][1]); trBG_hi = Fr(R['trBG'][1])
    EF = Fr(R['E_frob_upper']); DF = Fr(R['Ld_minus_Lt_frob_upper']); LdF = Fr(R['Ld_frob_upper'])
    e_ell = Fr(R['ell_table_error_upper'])
    n = S['n']
    gmax = sqrt_up(Fr(S['max_gnorm2']) * (1 + 4 * u))
    rho = Fr(S['max_absReD'])
    assert 2 * rho + 120 < 700, 'possible exp underflow: budget below not valid'

    # --- floating-point budget per sample (see README)
    epsD = (DF + (gam(m - 1) + 2 * u) * LdF * (1 + u)) * gmax          # ||computed D - Lt g||_2
    dw = e_ell + Fr(301, 100) * u * (rho + 120) + 16 * u                 # per-coordinate log perturbation
    eps_lip = 8 * M * (epsD + sqrt_up(m) * dw)
    eps_eval_one = Fr(101, 100) * (m * alpha * (3 * gam(m) + 5 * gam(k * k + 3)) + M * (gam(m + 2) + 4 * u))
    eps = eps_lip + 2 * 2 * eps_eval_one                                  # extra factor 2 safety
    # --- exact sums, with the rounding of math.fsum accounted
    sX = Fr(*S['sX']) + Fr(S['sX_err']); sX2 = Fr(*S['sX2']) + Fr(S['sX2_err'])
    sabs = Fr(*S['sabsX']) + Fr(S['sabsX_err'])
    meanX = sX / n + eps
    m2 = sX2 / n + 2 * eps * sabs / n + eps * eps
    # --- stage 1: E X^2 <= sbar, |X| <= 2M
    l1 = ln_up(10 ** kd); l2 = l1
    Rb = 2 * M
    kap = sqrt_up(2 * l1 / n)
    x = (Rb * kap + sqrt_up(Rb * Rb * kap * kap + 4 * m2)) / 2
    sbar = x * x
    # --- stage 2: one-sided Bernstein, Var X <= sbar, mu - X <= b = 2M
    b = 2 * M
    a = 2 * l2 * b / 3
    ustat = (a + sqrt_up(a * a + 8 * n * l2 * sbar)) / (2 * n)
    # --- covariance coupling
    W2 = sqrt_up(2 * sqrt_up(m - 1) * EF)
    coup = 8 * M * W2
    N_up = meanX + ustat + coup + trBG_hi
    thr = num_lo / pi_hi
    out = dict(n=n, failure_probability_upper=f'2e-{kd}', eps_per_sample=float(eps),
               mean_X_upper=float(meanX), second_moment_upper=float(m2), sbar=float(sbar),
               u_stat=float(ustat), coupling=float(coup), N_upper=float(N_up), threshold=float(thr),
               margin=float(thr - N_up), certified=bool(N_up < thr),
               ratio_lower=float(num_lo / N_up))
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, 'certificate.json'), 'w'), indent=1)


if __name__ == '__main__':
    main()
