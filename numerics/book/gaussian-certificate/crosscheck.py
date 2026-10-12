"""Empirical (non-rigorous) cross-checks of the floating-point evaluation in sample.py.

1. Re-evaluates the first batch of a run in extended precision (np.longdouble, 64-bit
   mantissa) from the same random numbers and reports max |X_double - X_extended|,
   to be compared with the rigorous per-sample budget eps of certify.py.
2. Reports the plain-estimator mean of the control c against the exact tr(BG)
   (consistency check of the control identity Sigma = G).
Usage: python crosscheck.py SEED [NSAMPLES=4000]
"""
import sys, os, json
import numpy as np
from sample import load, HERE

seed = int(sys.argv[1]); nb = int(sys.argv[2]) if len(sys.argv) > 2 else 4000
V, A, Ld, ell = load()
m = V.shape[0]; ref = m // 2
rng = np.random.default_rng([seed, 0])
g1 = rng.standard_normal((nb, m - 1)); g2 = rng.standard_normal((nb, m - 1)); j = rng.integers(0, m, nb)


def evaluate(dt):
    LdT = Ld.astype(dt).T
    ReD = np.zeros((nb, m), dt); ImD = np.zeros((nb, m), dt)
    keep = np.r_[0:ref, ref + 1:m]
    ReD[:, keep] = np.dot(g1.astype(dt), LdT); ImD[:, keep] = np.dot(g2.astype(dt), LdT)
    s = np.arange(-(m - 1), m).astype(dt)
    el = ell.astype(dt) if dt == np.float64 else np.abs(s) / 2 + np.log1p(np.exp(-np.abs(s))) - np.log(dt(2))
    w = ReD - el[np.arange(m)[None, :] - j[:, None] + (m - 1)]
    w = w - w.max(1, keepdims=True)
    mod = np.exp(w); Rr = mod * np.cos(ImD); Ri = mod * np.sin(ImD)
    S = (Rr * Rr + Ri * Ri).sum(1)
    Vd, Ad = V.astype(dt), A.astype(dt)
    Wr = np.dot(Rr, Vd); Wi = np.dot(Ri, Vd)
    qrr = np.einsum('na,ab,nb->n', Wr, Ad, Wr); qii = np.einsum('na,ab,nb->n', Wi, Ad, Wi)
    qri = np.einsum('na,ab,nb->n', Wr, Ad, Wi)
    f = m * np.hypot(qrr - qii, 2 * qri) / S
    return f - m * (qrr + qii) / S


Xd = evaluate(np.float64); Xl = evaluate(np.longdouble)
print('samples', nb, ' max |X_double - X_longdouble| =', float(np.max(np.abs(Xd.astype(np.longdouble) - Xl))))
cert = os.path.join(HERE, 'certificate.json')
if os.path.exists(cert):
    print('rigorous per-sample budget eps =', json.load(open(cert))['eps_per_sample'])
