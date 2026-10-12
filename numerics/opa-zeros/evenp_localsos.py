"""Numerical (not a proof). SOS side of the local ceiling (companion of evenp_local.py).

Maximise k such that  Re g_p(h) - k Im g_p(h) - sum_{m=2}^{p} Re(F_m h^m)  is SOS in (x, y)
(basis: monomials of degree 1..p/2).  Then w_SOS = k/sqrt(1+k^2) <= cos theta_trunc,
and evenp_local.py gives cos theta_trunc <= LP value; the two bracket the local ceiling.
Prints the optimal F_m and the Gram matrix rank.   Usage: evenp_localsos.py p
"""
import sys
import numpy as np
import sympy as sp
import cvxpy as cp

p = int(sys.argv[1]) if len(sys.argv) > 1 else 6
k2 = p // 2
x, y = sp.symbols("x y", real=True)
h, hb = x + sp.I * y, x - sp.I * y
g = sp.expand((1 - (1 - h) ** k2 * (1 - hb) ** (k2 - 1)) * hb)
mons = [(i, t - i) for t in range(1, k2 + 1) for i in range(t, -1, -1)]
allm = sorted({(a[0] + b[0], a[1] + b[1]) for a in mons for b in mons})
idx = {m: j for j, m in enumerate(allm)}


def vec(e):
    v = np.zeros(len(allm))
    for (i, j), c in sp.Poly(sp.expand(e), x, y).terms():
        v[idx[(i, j)]] = float(c)
    return v


vR, vI = vec(sp.re(g)), vec(sp.im(g))
hm = {m: (vec(sp.re(sp.expand(h ** m))), vec(sp.im(sp.expand(h ** m)))) for m in range(2, p + 1)}
nb = len(mons)
A = np.zeros((len(allm), nb, nb))
for a in range(nb):
    for b in range(nb):
        A[idx[(mons[a][0] + mons[b][0], mons[a][1] + mons[b][1])], a, b] = 1
k = cp.Variable()
Fr = {m: cp.Variable() for m in hm}
Fi = {m: cp.Variable() for m in hm}
Q = cp.Variable((nb, nb), symmetric=True)
coef = vR - k * vI - sum(Fr[m] * hm[m][0] - Fi[m] * hm[m][1] for m in hm)
cons = [Q >> 0] + [cp.sum(cp.multiply(A[r], Q)) == coef[r] for r in range(len(allm))]
prob = cp.Problem(cp.Maximize(k), cons)
prob.solve(solver=cp.CLARABEL)
kv = k.value
print(f"p={p}: k_SOS = {kv:.9f}, w_SOS = {kv / np.sqrt(1 + kv * kv):.7f}")
print("F_m:", {m: complex(round(float(Fr[m].value), 6), round(float(Fi[m].value), 6)) for m in hm})
ev = np.linalg.eigvalsh(Q.value)
print("Gram eigenvalues:", np.array2string(ev, precision=3))

# zero set of the SOS remainder (approximate atoms of the extremal truncated measure),
# then the LP of evenp_local.py on discs of radius 0.03 around them (+ h -> 0 directions):
from scipy.optimize import minimize, linprog
kk = p // 2
F = {m: complex(float(Fr[m].value), float(Fi[m].value)) for m in hm}


def gnum(z):
    return (1 - (1 - z) ** kk * np.conj(1 - z) ** (kk - 1)) * np.conj(z)


def rem(z):
    return (gnum(z).real - kv * gnum(z).imag - sum((F[m] * z ** m).real for m in F)) / abs(z) ** 2


XX, YY = np.meshgrid(np.linspace(-3, 3, 601), np.linspace(-3, 3, 601))
Z = (XX + 1j * YY).ravel()
Z = Z[abs(Z) > 0.05]
R = rem(Z)
atoms = []
for i in np.argsort(R)[:400]:
    r = minimize(lambda v: rem(v[0] + 1j * v[1]), [Z[i].real, Z[i].imag], method="Nelder-Mead",
                 options={"xatol": 1e-10, "fatol": 1e-14})
    zz = r.x[0] + 1j * r.x[1]
    if r.fun < 1e-5 and all(abs(zz - a) > 1e-4 for a in atoms):
        atoms.append(zz)
print("min rem on grid:", R.min(), Z[np.argmin(R)]); print("zeros of remainder (|h|>0.05):", np.round(atoms, 5))
small = 1e-4 * np.exp(2j * np.pi * np.arange(720) / 720)
loc = np.array([a + 0.03 * u * np.exp(1j * v) for a in atoms for u in np.linspace(0, 1, 15)
                for v in np.linspace(0, 2 * np.pi, 24, endpoint=False)])
pts = np.concatenate([loc, small])
sc = 1 / abs(pts) ** 2
rows = []
for m in range(2, p + 1):
    v = pts ** m * sc
    rows += [v.real, v.imag]
gv = gnum(pts) * sc
res = linprog(-gv.imag, A_eq=np.array(rows + [gv.real]), b_eq=np.r_[np.zeros(len(rows)), 1],
              bounds=(0, None), method="highs")
tl = -res.fun
print(f"LP on these atoms: tan = {tl:.9f}, cos theta = {np.cos(np.arctan(tl)):.7f}"
      f"  (SOS: {kv / np.sqrt(1 + kv * kv):.7f})")
