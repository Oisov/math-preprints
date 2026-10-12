"""Numerical (not a proof). Certificate with omega equal to the closing floor
4(1+delta)c1(sigma-1)^2/sigma, coefficients rounded to rationals with denominator 1000,
then a dense (theta, h) positivity test.  Usage: sos_rational.py p d nth mmax w
"""
import sys
from fractions import Fraction
import numpy as np
import sos_cert as S

w = float(sys.argv[5])
ok, (c1, f, om) = S.feasible(w, tight=True)
q = lambda v: Fraction(round(float(v) * 1000), 1000)
c1r = q(c1)
fr = {k: q(v.value) for k, v in f.items()}
print("SDP ok", ok, " c1 =", c1r, " f_{m,n} =", {k: str(v) for k, v in fr.items() if v})
th = np.linspace(0, np.pi, 3001)
s = 1 / (1 - w * np.exp(-1j * th))
omd = 4 * (1 + S.delta) * float(c1r) * (s.real - 1) ** 2 / s.real
r = np.concatenate([np.geomspace(1e-3, 1, 60), np.linspace(1, 4, 150), np.geomspace(4, 100, 40)])
a = np.linspace(-np.pi, np.pi, 721)
R, A = np.meshgrid(r, a)
h = (R * np.exp(1j * A)).ravel()
E = 1 - h
g = (1 - np.abs(E) ** (S.p - 2) * E) * np.conj(h)
worst = np.inf
for k in range(len(th)):
    sk = s[k]
    val = np.real(g * sk) - float(c1r) * np.real(sk * h * h) - omd[k] * h.imag ** 2
    for (m, n), v in fr.items():
        val -= float(v) * np.real(np.conj(sk) ** n * h ** m)
    worst = min(worst, (val / np.abs(h) ** 2).min())
print(f"p={S.p} w={w}: dense min of rounded certificate/|h|^2 = {worst:.3e}")
