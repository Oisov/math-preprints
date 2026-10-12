"""Numerical (not a proof). Solve the joint SDP of sos_cert.py at one w, then test the
certificate off the sample: on a dense theta grid (omega interpolated by the closing bound
4(1+delta)c1(sigma-1)^2/sigma plus the SDP excess, linearly) and a dense polar h-grid,
report min of the certificate polynomial divided by |h|^2.
Usage: sos_check.py p d nth mmax w
"""
import sys
import numpy as np
import sos_cert as S

w = float(sys.argv[5])
ok, (c1, f, om) = S.feasible(w, verbose=True)
th_s = np.pi * (np.arange(S.nth) + 0.5) / S.nth
s_s = 1 / (1 - w * np.exp(-1j * th_s))
floor_s = 4 * (1 + S.delta) * c1 * (s_s.real - 1) ** 2 / s_s.real
ex = om - floor_s
th = np.linspace(0, np.pi, 4001)
s = 1 / (1 - w * np.exp(-1j * th))
omd = 4 * (1 + S.delta) * c1 * (s.real - 1) ** 2 / s.real + np.maximum(np.interp(th, th_s, ex), 0) * 0.999
r = np.concatenate([np.geomspace(1e-3, 1, 60), np.linspace(1, 4, 150), np.geomspace(4, 100, 40)])
a = np.linspace(-np.pi, np.pi, 721)
R, A = np.meshgrid(r, a)
h = (R * np.exp(1j * A)).ravel()
E = 1 - h
g = (1 - np.abs(E) ** (S.p - 2) * E) * np.conj(h)
worst = np.inf
for k in range(len(th)):
    sk = s[k]
    val = np.real(g * sk) - c1 * np.real(sk * h * h) - omd[k] * h.imag ** 2
    for (m, n), v in f.items():
        val -= float(v.value) * np.real(np.conj(sk) ** n * h ** m)
    worst = min(worst, (val / np.abs(h) ** 2).min())
print(f"p={S.p} w={w}: SDP ok={ok}, c1={c1:.6f}, dense min of certificate/|h|^2 = {worst:.3e}")
