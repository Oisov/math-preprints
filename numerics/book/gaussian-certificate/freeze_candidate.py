"""Freeze the certified candidate as exact binary64 data (hex strings).

Source: re-optimized span-240, 6-mode candidate of
../code/critical-kernel-independent/opt_T240_k6_h1.json (coefficients A in the
basis V = top-6 G-orthonormal generalized eigenvectors of (H, G), grid
t = -120..120, spacing 1).  The basis is recomputed here and stored; from now on
the candidate *is* the stored binary64 numbers (exact rationals), whatever their
origin.  Usage: python freeze_candidate.py
"""
import json, os
import numpy as np
from scipy.linalg import eigh

HERE = os.path.dirname(os.path.abspath(__file__))
src = json.load(open(os.path.join(HERE, '..', 'code', 'critical-kernel-independent', 'opt_T240_k6_h1.json')))
m, k = src['m'], src['k']
t = np.arange(m) - (m - 1) // 2                      # integers -120..120
d = (t[:, None] - t[None, :]).astype(float)
G = 1 / np.cosh(d / 2)
with np.errstate(invalid='ignore', divide='ignore'):
    H = np.where(d == 0, 2.0, d / np.sinh(d / 2))
w, U = eigh(H, G)
V = U[:, ::-1][:, :k]
assert np.allclose(w[::-1][:k], src['lam'], rtol=0, atol=1e-12)
A = np.array(src['A'])
A = (A + A.T) / 2
out = dict(t=[int(x) for x in t], k=k,
           V=[[float(x).hex() for x in row] for row in V],
           A=[[float(x).hex() for x in row] for row in A],
           note='Exact binary64 data; integrand uses B = V A V^T, reference scale t=0.')
json.dump(out, open(os.path.join(HERE, 'candidate.json'), 'w'))
print('frozen m=%d k=%d, tr A=%.6f' % (m, k, np.trace(A)))
