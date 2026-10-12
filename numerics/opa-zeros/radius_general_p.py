"""Exploratory (numerical, not a proof): does the p=4 certificate of Theorem 5 of the paper
extend to other p?

Ansatz: for s = S_w(e^{i theta}), real alpha, beta,
   Re(g_p(h) s) - Re((alpha s + beta conj s) h^2) >= omega(theta) (Im h)^2  for all h,
with g_p(h) = (1 - |1-h|^{p-2}(1-h)) conj h, and then
   int omega y^2 + Re(alpha H(w)^2) > 0  for all H(0)=0,  i.e.  4 max(alpha G11, -alpha G22) < 1,
G11 = min_c int (sigma - c)^2 / omega dm, G22 = int tau^2 / omega dm (s = sigma + i tau).
The infimum defining omega is taken over a finite h-grid, so this is exploratory only.  If this holds, Re Pi_H(w) > 0 and |w| is not a zero radius.
For each p we bisect on w and report the largest certified w next to the two-pole value.
"""
import sys
import numpy as np
from scipy.optimize import minimize

p = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
# h grid in polar coordinates (dense near 0 and moderate radii)
rad = np.concatenate([np.geomspace(1e-3, 0.5, 40), np.linspace(0.5, 6, 130), np.geomspace(6, 60, 30)])
ang = np.linspace(-np.pi, np.pi, 361)[:-1]
R, A = np.meshgrid(rad, ang)
h = (R * np.exp(1j * A)).ravel()
E = 1 - h
g = (1 - np.abs(E) ** (p - 2) * E) * np.conj(h)
y2 = h.imag ** 2
realmask = y2 < 1e-12 * np.abs(h) ** 2
nth = 120
th = np.pi * (np.arange(nth) + 0.5) / nth


def margin(w, ab):
    al, be = ab
    s = 1 / (1 - w * np.exp(-1j * th))
    om = np.empty(nth)
    for k, sk in enumerate(s):
        num = np.real(g * sk) - np.real((al * sk + be * np.conj(sk)) * h * h)
        if num[realmask].min() < -1e-12:
            return -1.0
        q = num[~realmask] / y2[~realmask]
        om[k] = q.min()
    if om.min() <= 0:
        return -1.0
    sig = s.real
    I0, I1, I2 = np.mean(1 / om), np.mean(sig / om), np.mean(sig ** 2 / om)
    G11 = I2 - I1 ** 2 / I0
    G22 = np.mean(s.imag ** 2 / om)  # tau is odd in theta, omega even
    return 1 - 4 * max(al * G11, -al * G22)


def best(w):
    bestv = -9
    for x0 in [(0.25 * (p - 2) / 2, 0.5 * (p - 2) / 2), ((p - 2) / 8, (p - 2) / 2), (0.0, (p - 2) / 2)]:
        r = minimize(lambda x: -margin(w, x), x0, method="Nelder-Mead",
                     options={"xatol": 1e-6, "fatol": 1e-8, "maxiter": 150})
        bestv = max(bestv, -r.fun)
    return bestv


lo, hi = 0.85, 0.9999
for _ in range(12):
    mid = (lo + hi) / 2
    if best(mid) > 0:
        lo = mid
    else:
        hi = mid
print(f"p={p}: ansatz certifies no zeros for |w| <= {lo:.6f}")
