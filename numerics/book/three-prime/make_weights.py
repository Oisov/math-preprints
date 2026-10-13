"""Regenerate schur_weights.json for pi3_upper.py (Theorem 9.15).

The Schur weights are the moduli of the leading singular vectors of the
truncated matrix T = (c_{mn})_{mn <= N0}, c_N = 1/(sqrt(N) log N), computed in
binary64 with numpy, with entries below 1e-12 times the maximum raised to that
value.  They are stored as exact hexadecimal binary64 numbers; pi3_upper.py
reads them as exact dyadic rationals and does not trust how they were found.

Run: python3 make_weights.py   (needs numpy; about 10 s)
"""
import json
from pathlib import Path

import numpy as np

from pi3_upper import N0, smooth_numbers

S = smooth_numbers(N0)
r = np.array(S, dtype=float)
c = r[1:]
NN = np.outer(r, c)
A = np.where(NN <= N0, 1.0 / (np.sqrt(NN) * np.log(NN)), 0.0)
U, Sv, Vt = np.linalg.svd(A, full_matrices=False)
u = np.abs(U[:, 0])
v = np.abs(Vt[0])
u = np.maximum(u, 1e-12 * u.max())
v = np.maximum(v, 1e-12 * v.max())
Path(__file__).with_name("schur_weights.json").write_text(json.dumps({
    "N0": N0,
    "float_singular_value": Sv[0].hex(),
    "u_rows_hex": [x.hex() for x in u],
    "v_cols_hex": [x.hex() for x in v],
}) + "\n")
print("largest singular value (binary64):", Sv[0])
