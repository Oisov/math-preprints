# Numerics for *Remarks on three sharpness questions for composition operators on spaces of Dirichlet series*

Paper: "Remarks on three sharpness questions for composition operators on spaces of Dirichlet series" (`papers/composition-sharpness.pdf`).

| Statement | Claim | Script | Arithmetic |
|---|---|---|---|
| Lemma 7 (finite jets) | the displayed Taylor-coefficient identities for `[f1]_3`, `[f1]_4`, `[f3]_2`, `[f3]_3`, `[f3]_4`, the determinant, and the reductions modulo `S^3 ω^2 + V` | `jets.py` | exact, over Q(a1,a2,x1,x2,ω) (sympy) |

Run `python3 jets.py`; it prints `ALL OK` (about one second). Requires
`sympy` (tested with 1.13.3). The relations "modulo `S^3 ω^2 + V`" are checked
by polynomial division in ω, so no square roots or floating point occur.

The script is the only computer-assisted step in the paper; the rest of the
proof of Theorem 3 is by hand.
