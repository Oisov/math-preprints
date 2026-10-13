"""Certificate for Proposition 8.10 (and the constants of Proposition 8.9).

Notation of Section 8: X = R_n^2 is a product of n independent Exp(1)
variables, A = E(1-k^2 X)_+, K = E[R min(kR,1)], eps_n = E R = Gamma(3/2)^n,
Phi = A^2 K + 2^(n-1) K^3 - k A^3 - 2 k A K^2.  Proposition 8.10 claims Phi < 0
for 194 <= n <= 3998 and k = exp(kappa n), kappa = 0.4276.

Part 1 (194 <= n <= 3998).  With s = c + it, c = 3/20, x = 1 - c,
    A = (1/2pi) int Re[k^(-2s) Gamma(1-s)^n / (s(s+1))] dt.
Factor out S_n = k^(-2c) Gamma(x)^n = exp(n lS), lS = log Gamma(x) - 2 c kappa:
    A = S_n a_n,  a_n = (1/2pi) int Re[exp(n h(t)) / (s(s+1))] dt,
    h(t) = log Gamma(1-s) - log Gamma(x) - 2 i kappa t.
The part |t| <= 2 is enclosed by rigorous Gauss-Legendre integration in ball
arithmetic; the integrand is declared non-holomorphic where Re(1-s) <= 1/20.
The part |t| > 2 is at most 2 (1 + 4/x^2)^(-n/2) / 2 (not divided by 2pi),
from |Gamma(x+iy)| <= Gamma(x) (1 + y^2/x^2)^(-1/2) and |s(s+1)| >= t^2.
With K_- = eps_n - 1/(4k) = eps_n (1 - delta), delta = exp(-n(kappa + lE))/4,
lE = log Gamma(3/2), the bound of Lemma 8.8,
    Phi <= A_+^2 eps_n + 2^(n-1) eps_n^3 - k A_-^3 - 2 k A_- K_-^2,
divided by 2^(n-1) K_-^3 > 0 becomes
    R = [2 e^(n(2lS-log2-2lE)) a_+^2 + 1 - 2 e^(n(kappa+3lS-log2-3lE)) a_-^3
         - 4 e^(n(kappa+lS-log2-lE)) a_- (1-delta)^2] / (1-delta)^3,
and Phi < 0 follows from R < 0.  The script checks the stronger R <= -0.01165,
the value printed in the book as the largest certified ratio (attained at
n = 194, where R is about -0.011656).  Every quantity is a continuous expression in n,
so the script evaluates it with n replaced by the ball [n1, n2] for each block
of consecutive integers listed in clipped_blocks.json: the resulting enclosure
contains R for every integer n of the block.  The blocks are single integers
near n = 194, where the margin is small, and widen as the margin grows
exponentially.  They were found by make_blocks.py; this script only checks them
and that they cover 194..3998 exactly.

Part 2 (n >= 3999, Proposition 8.9).  Checks rho > 0 and
n rho > c1 sqrt(n) + c0 at the smallest such integer N (= 3999), with s = 3/20,
which then holds for all n >= N since the left side minus the right is
increasing for sqrt(n) > c1/(2 rho).

Run: python3 clipped_finite.py   (needs python-flint)
"""
import json
import sys
from pathlib import Path

from flint import acb, arb, ctx

ctx.prec = 64
C = arb(3) / 20
X = 1 - C
KAPPA = arb(4276) / 10000
T = 2
TOL = arb(2) ** -14
LGX = X.lgamma()
LE = (arb(3) / 2).lgamma()
LOG2 = arb(2).log()
LS = LGX - 2 * C * KAPPA
THRESHOLD = arb("-0.01165")
BLOCKS = Path(__file__).resolve().parent / "clipped_blocks.json"


def ratio(n1, n2):
    """Enclosure of R containing its value at every integer n in [n1, n2]."""
    n = arb(n1).union(arb(n2))
    I1 = acb(0, 1)

    def f(z, analytic):
        s = acb(C) + I1 * z
        w = 1 - s
        if analytic and not (w.real > arb(1) / 20):
            return acb(float("nan"))
        return (n * (w.lgamma() - LGX - 2 * KAPPA * I1 * z)).exp() / (s * (s + 1))

    integral = acb.integral(f, -T, T, rel_tol=TOL, abs_tol=TOL)
    tail = 2 * (1 + arb(T) ** 2 / X ** 2) ** (-n / 2) / arb(T)
    a = integral.real / (2 * arb.pi()) + arb(0, tail.upper())
    a_lo, a_hi = arb(a.lower()), arb(a.upper())
    delta = (-n * (KAPPA + LE)).exp() / 4
    if not (a_lo > 0 and delta < 1):
        return arb(0, 1e300)
    t1 = 2 * (n * (2 * LS - LOG2 - 2 * LE)).exp() * a_hi ** 2
    t3 = 2 * (n * (KAPPA + 3 * LS - LOG2 - 3 * LE)).exp() * a_lo ** 3
    t4 = 4 * (n * (KAPPA + LS - LOG2 - LE)).exp() * a_lo * (1 - delta) ** 2
    return (t1 + 1 - t3 - t4) / (1 - delta) ** 3


def part2():
    ctx.prec = 128
    s = arb(3) / 20
    x = 1 - s
    psi, psi1 = x.digamma(), arb(2).zeta(x)  # psi'(x) = zeta(2, x)
    rho = (-psi / 2 + 3 * x.lgamma() + 3 * s * psi
           - arb(2).log() - 3 * (arb(3) / 2).gamma().log())
    c1 = (arb(1) / 2 + 3 * s) * (2 * psi1).sqrt()
    c0 = arb(1) / 2 - 3 * ((1 - arb(-1).exp()) / 2).log()
    assert rho > 0
    root = (c1 + (c1 ** 2 + 4 * rho * c0).sqrt()) / (2 * rho)
    N = int((root ** 2).upper().floor().unique_fmpz()) + 1
    g = arb(N) * rho - c1 * arb(N).sqrt() - c0
    assert g > 0 and arb(N).sqrt() > c1 / (2 * rho)
    ctx.prec = 64
    return N, rho, c1, c0, g


def main():
    N, rho, c1, c0, g = part2()
    print(f"part 2: s=3/20, rho={rho.str(8)}, c1={c1.str(8)}, "
          f"c0={c0.str(8)}, N={N}, margin at N={g.str(5)}")
    assert N == 3999
    blocks = json.loads(BLOCKS.read_text())["blocks"]
    expect = 194
    worst = None
    for n1, n2 in blocks:
        assert n1 == expect and n2 >= n1
        r = ratio(n1, n2)
        if not r < THRESHOLD:
            print(f"FAIL on block [{n1}, {n2}]: ratio {r}")
            sys.exit(1)
        if worst is None or r.upper() > worst[1].upper():
            worst = (n1, r)
        expect = n2 + 1
    assert expect == N
    print(f"part 1: Phi_n(exp(0.4276 n)) < 0 certified for 194 <= n <= {N - 1} "
          f"on {len(blocks)} blocks; largest ratio Phi/(2^(n-1)K_-^3) bound "
          f"{float(worst[1].upper()):.6f} at n={worst[0]}")
    print(f"n=193 at the same k: ratio {ratio(193, 193).str(6)} (not negative)")


if __name__ == "__main__":
    main()
