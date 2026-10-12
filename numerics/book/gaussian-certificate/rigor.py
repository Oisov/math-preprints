"""Deterministic, exact/interval part of the certificate.

All transcendental constants are enclosed with fixed-point interval arithmetic on
Python integers (scale 2^-P, outward rounding); the candidate data are exact
binary64 rationals.  Outputs rigor.json (interval endpoints as decimal strings
rounded outward), Ld.npy (the binary64 Gaussian factor used by the sampler) and
ell.npy (the binary64 table of log cosh(s/2) used by the sampler).

Usage: python rigor.py
"""
import json, os, math, time
from fractions import Fraction as Fr
from decimal import Decimal, getcontext
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
P = 640                       # fixed-point bits
ONE = 1 << P


def fl(a, b):  # floor(a/b), b>0
    return a // b


def cl(a, b):
    return -((-a) // b)


class I:
    """Interval [lo, hi] * 2^-P with integer endpoints."""
    __slots__ = ('lo', 'hi')

    def __init__(self, lo, hi=None):
        self.lo, self.hi = lo, (lo if hi is None else hi)
        assert self.lo <= self.hi

    @staticmethod
    def q(x):  # exact rational -> enclosing interval
        x = Fr(x)
        return I(fl(x.numerator * ONE, x.denominator), cl(x.numerator * ONE, x.denominator))

    def __add__(s, o):
        o = o if isinstance(o, I) else I.q(o)
        return I(s.lo + o.lo, s.hi + o.hi)
    __radd__ = __add__

    def __neg__(s):
        return I(-s.hi, -s.lo)

    def __sub__(s, o):
        o = o if isinstance(o, I) else I.q(o)
        return s + (-o)

    def __rsub__(s, o):
        return (-s) + o

    def __mul__(s, o):
        if not isinstance(o, I):
            o = Fr(o)
            ps = [s.lo * o.numerator, s.hi * o.numerator]
            d = o.denominator
            return I(fl(min(ps), d), cl(max(ps), d))
        ps = [s.lo * o.lo, s.lo * o.hi, s.hi * o.lo, s.hi * o.hi]
        return I(fl(min(ps), ONE), cl(max(ps), ONE))
    __rmul__ = __mul__

    def __truediv__(s, o):
        o = o if isinstance(o, I) else I.q(o)
        assert o.lo > 0
        qs = [s.lo * ONE, s.hi * ONE]
        return I(min(fl(x, y) for x in qs for y in (o.lo, o.hi)),
                 max(cl(x, y) for x in qs for y in (o.lo, o.hi)))

    def __rtruediv__(s, o):
        return I.q(o) / s

    def flo(s):
        return Fr(s.lo, ONE)

    def fhi(s):
        return Fr(s.hi, ONE)

    def mid(s):
        return Fr(s.lo + s.hi, 2 * ONE)

    def width(s):
        return Fr(s.hi - s.lo, ONE)


def exp_half():
    """e^{1/2}: Taylor partial sum plus remainder <= 2 (1/2)^{K+1}/(K+1)!."""
    K, s, term = 200, Fr(0), Fr(1)
    for k in range(K + 1):
        s += term
        term = term / 2 / (k + 1)
    return I(fl(s.numerator * ONE, s.denominator), cl((s + 2 * term).numerator * ONE, (s + 2 * term).denominator))


def atanh_series(z):
    """atanh(z) for an interval 0 <= z <= 1/2: partial sum + geometric tail bound."""
    zl, zh = z.flo(), z.fhi()
    assert 0 <= zl and zh <= Fr(1, 2)
    K = 1
    while zh ** (2 * K + 1) > Fr(1, 2 ** (P + 20)) and K < 2000:
        K += 1
    z2 = z * z
    s, pw = I(0), z
    for k in range(K):
        s = s + pw * Fr(1, 2 * k + 1)
        pw = pw * z2
    tail = zh ** (2 * K + 1) / ((2 * K + 1) * (1 - zh * zh))
    return I(s.lo, s.hi + cl(tail.numerator * ONE, tail.denominator))


def atan_inv(n):
    """atan(1/n), alternating series, n>=2."""
    s = Fr(0)
    K = 0
    while Fr(1, n ** (2 * K + 1)) > Fr(1, 2 ** (P + 20)):
        K += 1
    for k in range(K):
        s += Fr((-1) ** k, (2 * k + 1) * n ** (2 * k + 1))
    r = Fr(1, (2 * K + 1) * n ** (2 * K + 1))
    lo, hi = (s, s + r) if K % 2 == 0 else (s - r, s)
    return I(fl(lo.numerator * ONE, lo.denominator), cl(hi.numerator * ONE, hi.denominator))


def dec(x, up):
    """Fraction -> decimal string with 40 digits, rounded outward."""
    x = Fr(x)
    sc = 10 ** 40
    v = cl(x.numerator * sc, x.denominator) if up else fl(x.numerator * sc, x.denominator)
    return str(Decimal(v) / Decimal(sc))


def sqrt_up(x):
    """Upper bound for sqrt of a nonnegative Fraction."""
    x = Fr(x)
    if x == 0:
        return Fr(0)
    sc = 1 << 200
    r = math.isqrt(cl(x.numerator * sc * sc, x.denominator)) + 1
    return Fr(r, sc)


def main():
    t0 = time.time()
    cand = json.load(open(os.path.join(HERE, 'candidate.json')))
    t = cand['t']; m = len(t); k = cand['k']
    V = [[float.fromhex(x) for x in row] for row in cand['V']]
    A = [[Fr(float.fromhex(x)) for x in row] for row in cand['A']]
    assert all(A[a][b] == A[b][a] for a in range(k) for b in range(k))
    assert t == list(range(-(m // 2), m // 2 + 1))
    ref = t.index(0)

    # --- constants
    eh = exp_half()
    emh = 1 / eh
    pi = atan_inv(5) * 16 - atan_inv(239) * 4
    log2 = atanh_series(I.q(Fr(1, 3))) * 2
    epow = [I(ONE)]
    empow = [I(ONE)]
    for d in range(1, m):
        epow.append(epow[-1] * eh)
        empow.append(empow[-1] * emh)
    g = [I(ONE)] + [2 / (epow[d] + empow[d]) for d in range(1, m)]          # sech(d/2)
    h = [I(2 * ONE)] + [(2 * d) / (epow[d] - empow[d]) for d in range(1, m)]  # d/sinh(d/2)
    # L(d) = log cosh(d/2) = d/2 - log 2 + log(1+e^{-d}),  log(1+x) = 2 atanh(x/(2+x))
    Lc = []
    for d in range(m):
        x = empow[d] * empow[d]
        Lc.append(I.q(Fr(d, 2)) - log2 + atanh_series(x / (x + 2)) * 2)
    print('constants', time.time() - t0)

    # --- exact Toeplitz correlations of V (integers after common scaling)
    mant = [[Fr(v) for v in row] for row in V]
    e0 = min(min(f.denominator.bit_length() for f in row) for row in mant)
    sh = max(max(f.denominator.bit_length() - 1 for f in row) for row in mant)
    Vi = [[int(f * (1 << sh)) for f in row] for row in mant]
    assert all(Fr(Vi[r][a], 1 << sh) == mant[r][a] for r in range(m) for a in range(k))
    # wH_d, wG_d: exact rationals with  num = sum_d h_d * w_d,  sum_ab A_ab (V^T T V)_ab = sum_d T_d w_d
    w = []
    for d in range(m):
        tot = Fr(0)
        for a in range(k):
            for b in range(k):
                if A[a][b] == 0:
                    continue
                c = sum(Vi[r + d][a] * Vi[r][b] for r in range(m - d))
                if d > 0:
                    c += sum(Vi[r + d][b] * Vi[r][a] for r in range(m - d))
                tot += A[a][b] * c
        w.append(tot / (1 << (2 * sh)))
    num = sum((h[d] * w[d] for d in range(m)), I(0))
    trBG = sum((g[d] * w[d] for d in range(m)), I(0))
    # ||B||_F^2 = tr(A P A P), P = V^T V
    Pm = [[Fr(sum(Vi[r][a] * Vi[r][b] for r in range(m)), 1 << (2 * sh)) for b in range(k)] for a in range(k)]
    AP = [[sum(A[a][c] * Pm[c][b] for c in range(k)) for b in range(k)] for a in range(k)]
    BF2 = sum(AP[a][b] * AP[b][a] for a in range(k) for b in range(k))
    M = m * sqrt_up(BF2)                   # sup of the integrand f and of |control|
    alpha = sum(abs(A[a][b]) * sqrt_up(Pm[a][a]) * sqrt_up(Pm[b][b]) for a in range(k) for b in range(k))
    target = num / pi                      # numerator / pi
    print('numerator', float(num.mid()), 'width', float(num.width()))
    print('trBG', float(trBG.mid()), ' M', float(M), ' num/pi', float(target.mid()))

    # --- Gaussian factor of C/2 on the non-reference coordinates
    idx = [i for i in range(m) if i != ref]
    n = len(idx)

    def Lof(d):
        return Lc[abs(d)]
    Chalf = [[(Lof(t[i]) + Lof(t[j]) - Lof(t[i] - t[j])) * Fr(1, 2) for j in idx] for i in idx]
    getcontext().prec = 110
    Cd = [[Decimal(x.lo + x.hi) / Decimal(2 * ONE) for x in row] for row in Chalf]
    Ldc = [[Decimal(0)] * n for _ in range(n)]
    for j in range(n):
        s = Cd[j][j] - sum(Ldc[j][q] * Ldc[j][q] for q in range(j))
        assert s > 0
        Ldc[j][j] = s.sqrt()
        for i in range(j + 1, n):
            Ldc[i][j] = (Cd[i][j] - sum(Ldc[i][q] * Ldc[j][q] for q in range(j))) / Ldc[j][j]
    print('decimal cholesky', time.time() - t0, 'min diag', float(min(Ldc[j][j] for j in range(n))))
    Lint = [[int((Ldc[i][j] * (1 << P)).to_integral_value()) for j in range(n)] for i in range(n)]
    Lo = np.array(Lint, dtype=object)
    LL = Lo.dot(Lo.T)                       # exact, scale 2^-2P
    E2 = Fr(0)
    for i in range(n):
        for j in range(n):
            e = Chalf[i][j] - I(fl(LL[i, j], ONE), cl(LL[i, j], ONE))
            E2 += Fr(max(abs(e.lo), abs(e.hi)), ONE) ** 2
    EF = sqrt_up(E2)                         # ||C/2 - Lt Lt^T||_F
    # binary64 factor and its exact distance to Lt = Lint/2^P
    Ld = np.array([[float(Fr(Lint[i][j], ONE)) for j in range(n)] for i in range(n)])
    D2 = sum((Fr(Ld[i, j]) - Fr(Lint[i][j], ONE)) ** 2 for i in range(n) for j in range(n))
    DF = sqrt_up(D2)
    LdF = sqrt_up(sum(Fr(x) ** 2 for x in Ld.ravel()))
    np.save(os.path.join(HERE, 'Ld.npy'), Ld)
    print('factor: ||E||_F <=', float(EF), ' ||Ld-Lt||_F <=', float(DF), ' ||Ld||_F <=', float(LdF), time.time() - t0)

    # --- binary64 table ell[s+m-1] ~ log cosh(s/2), and its exact error
    s = np.arange(-(m - 1), m).astype(float)
    ell = np.abs(s) / 2 + np.log1p(np.exp(-np.abs(s))) - np.log(2.0)
    np.save(os.path.join(HERE, 'ell.npy'), ell)
    e_ell = max(max(abs(Fr(float(ell[i])) - Lc[abs(int(s[i]))].flo()),
                    abs(Fr(float(ell[i])) - Lc[abs(int(s[i]))].fhi())) for i in range(len(s)))

    out = dict(m=m, k=k, ref_index=ref,
               numerator=[dec(num.flo(), 0), dec(num.fhi(), 1)],
               pi=[dec(pi.flo(), 0), dec(pi.fhi(), 1)],
               numerator_over_pi_lo=dec((num / pi).flo(), 0),
               trBG=[dec(trBG.flo(), 0), dec(trBG.fhi(), 1)],
               M_upper=dec(M, 1), alpha_upper=dec(alpha, 1),
               E_frob_upper=dec(EF, 1), Ld_minus_Lt_frob_upper=dec(DF, 1), Ld_frob_upper=dec(LdF, 1),
               ell_table_error_upper=dec(e_ell, 1),
               min_cholesky_diag=float(min(Ldc[j][j] for j in range(n))),
               sha_note='Ld.npy and ell.npy are the exact inputs of sample.py')
    json.dump(out, open(os.path.join(HERE, 'rigor.json'), 'w'), indent=1)
    print(json.dumps(out, indent=1))
    print('total', time.time() - t0)


if __name__ == '__main__':
    main()
