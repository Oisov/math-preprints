r"""Numerical (not a proof). Local ceiling of the pointwise certificate family.

At a tangency point (omega = 0) the certificate of the depth remark reads
  Re(g_p(h) s) - sum_{m=2}^{p} Re(F_m h^m) >= 0   for all h,
with F_m free complex numbers.  By LP duality it exists iff Re(s <g>_nu) >= 0 for every
positive measure nu on C\{0} with <h^m>_nu = 0, m = 2..p ("truncated representing
measures").  We compute  theta_trunc = sup |arg <g>_nu|  by an LP on a polar h-grid
(columns scaled by |h|^2, so the limit h -> 0 is included), and report cos(theta_trunc),
the largest |w| any certificate of this family can reach.  Compare with the profile
ceiling cos(theta_prof) (sos_profiles.py).  Usage: evenp_local.py p [nr] [nphi] [rmax]
"""
import sys
import numpy as np
from scipy.optimize import linprog

p = int(sys.argv[1]) if len(sys.argv) > 1 else 6
nr = int(sys.argv[2]) if len(sys.argv) > 2 else 120
nphi = int(sys.argv[3]) if len(sys.argv) > 3 else 360
rmax = float(sys.argv[4]) if len(sys.argv) > 4 else 6.0
k = p // 2


def g(h):
    E = 1 - h
    return (1 - E ** k * np.conj(E) ** (k - 1)) * np.conj(h)


r = np.concatenate([[1e-4], np.linspace(rmax / nr, rmax, nr)])
phi = 2 * np.pi * np.arange(nphi) / nphi
h = (r[:, None] * np.exp(1j * phi[None, :])).ravel()
sc = 1 / np.abs(h) ** 2
gv = g(h) * sc
rows, rhs = [], []
for m in range(2, p + 1):
    v = h ** m * sc
    rows += [v.real, v.imag]
    rhs += [0, 0]
rows.append(gv.real)
rhs.append(1)
res = linprog(-gv.imag, A_eq=np.array(rows), b_eq=np.array(rhs), bounds=(0, None), method="highs")
th = np.arctan(-res.fun)
print(f"p={p}: truncated-measure tan = {-res.fun:.8f}, cos(theta_trunc) = {np.cos(th):.6f}")
lam = res.x
sup = np.argsort(-lam)[:12]
for i in sup:
    if lam[i] > 1e-9:
        print(f"  atom h = {h[i]:.4f}  |h| = {abs(h[i]):.4f}  weight*|h|^-2 = {lam[i]:.4e}")
