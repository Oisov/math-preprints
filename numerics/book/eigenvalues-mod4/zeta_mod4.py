"""Certificate for Theorem 5.12 (and Corollary 5.13).

R = primes p = 1 (mod 4), theta = 1/2.  With P(s) = prod_{p = 3 (4)} (1 - p^{-2s})
and P = P(1),
    zeta_R(s)^2 = zeta(s) L(s, chi_4) (1 - 2^{-s}) P(s),   c_R^2 = pi P / 8.
The proof of Theorem 5.12 reduces zeta_R(1+t) - 1 <= c_R t^{-1/2} (t > 0) to

  (A) 1 - c_R^2 <= sqrt(2) c_R                                        (t >= 1/2),
  (B) f_L(1/2) <= sqrt(2) c_R,  f_L(t) = P (Q(t) e^{K t} - pi/8)       (0 < t <= 1/2),
      Q(t) = (1+t)(pi/4 + t log(3)/3)(1/2 + t log(2)/2),
      K = sum_{p = 3 (4)} 2 log p / (p^2 - 1).

Rigorous inputs (Arb balls at 128 bits), with X = 10^6:
  P <= prod_{p <= X} (1 - p^-2) =: P_X,   P >= P_X (1 - 1/X)
      (since sum_{n > X} n^-2 <= 1/X);
  K <= sum_{p <= X} 2 log p / (p^2 - 1) + (2 X^2 / (X^2 - 1)) (log X + 1) / X
      (since (log t)/t^2 decreases for t >= 2 and int_X^oo 2 log t / t^2 dt
       = 2 (log X + 1) / X).
The script also encloses c_R and L_R = c_R Gamma(1/4)^2 / sqrt(pi) and checks
the decimal bounds printed in Theorem 5.12 and its proof.

Run: python3 zeta_mod4.py   (needs python-flint; well under a second)
"""
from flint import arb, ctx

ctx.prec = 128
X = 10**6

sieve = bytearray([1]) * (X + 1)
sieve[0] = sieve[1] = 0
for i in range(2, int(X**0.5) + 1):
    if sieve[i]:
        sieve[i * i :: i] = bytearray(len(range(i * i, X + 1, i)))
primes3 = [p for p in range(3, X + 1, 4) if sieve[p]]

PX = arb(1)
Ksum = arb(0)
for p in primes3:
    PX *= 1 - arb(1) / (p * p)
    Ksum += 2 * arb(p).log() / (p * p - 1)
Xa = arb(X)
K = arb(Ksum.upper() + (2 * Xa**2 / (Xa**2 - 1) * (Xa.log() + 1) / Xa).upper())
P = arb(PX.upper()).union(arb((PX * (1 - 1 / Xa)).lower()))  # encloses P
pi = arb.pi()
c = (pi * P / 8).sqrt()
L = c * arb(0.25).gamma() ** 2 / pi.sqrt()

half = arb(1) / 2
Q = (1 + half) * (pi / 4 + half * arb(3).log() / 3) * (half + half * arb(2).log() / 2)
fL = arb(P.upper()) * (Q * (K * half).exp() - pi / 8)  # bracket > 0
sqrt2c = 2 * half.sqrt() * arb(c.lower())

print("P in [", P.lower(), ",", P.upper(), "]")
print("K <=", K.upper())
print("c_R in [", c.lower(), ",", c.upper(), "]")
print("L_R in [", L.lower(), ",", L.upper(), "]")
print("1 - c_R^2 <=", (1 - c**2).upper(), "  sqrt(2) c_R >=", sqrt2c.lower())
print("f_L(1/2) <=", fL.upper())

assert Q * (K * half).exp() > pi / 8
assert (1 - c**2) < sqrt2c, "(A) fails"
assert fL < sqrt2c, "(B) fails"
# the decimals printed in the book
assert arb("0.856108") < P and P < arb("0.856110")
assert K < arb("0.45751")
assert arb("0.579821") < c and c < arb("0.579822")
assert arb("4.30012") < L and L < arb("4.30014")
assert 1 - c**2 < arb("0.66382") and arb("0.81997") < sqrt2c
assert fL < arb("0.71642")
print("CERTIFIED: zeta_R(1+t) - 1 <= c_R t^(-1/2) for all t > 0")
