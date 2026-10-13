# Computations for *Where Nehari's Theorem Survives* (book draft, v4)

Every computer-assisted statement in the book, with the script that checks it.
Theorem numbers refer to the current draft (`book/where-nehari-survives-v4.pdf`).

`python3 run_all.py` runs the certificates in the table and compares their
output with the values printed in the book. The Gaussian probabilistic
certificate is not included, since it takes about 40 minutes on 16 cores; see
below for how to run it.

## Requirements

- Python 3.10 or later.
- `python-flint==0.9.0` (Arb ball arithmetic) for the scripts marked *Arb*:
  `python3 -m pip install --only-binary=:all: python-flint==0.9.0`.
- The C₀ scripts and `spectral-gap/floquet_gap.py` use only the standard library, with exact
  rational or directed dyadic arithmetic.
- The symbolic checks need `sympy`.
- The Gaussian certificate needs `numpy` and `scipy`.

## Certificates

| Book statement | Claim | Script | Arithmetic |
|---|---|---|---|
| Thm 2.8 (local constant), σ ≤ 1/100 | analytic margin at σ = 1/100 is about 0.0926 > 0 | `local-constant/small_sigma.py` | Arb |
| Thm 2.8 (local constant), 1/100 ≤ σ ≤ 2 | Schur rows below Γ(1/4)²/√(2π) on 268 cells; smallest margin 0.1769 | `local-constant/cells.py 0.01 2 1.02 1.005` | Arb |
| Prop 3.7(3) (constant at p = 2) | C₀ > 2π + 2.875·10⁻⁷ | `C0/lower.py` (reads `C0/witness.json`) | exact dyadic |
| Prop 3.7(3) | C₀ < 2π + 0.000973 < 2π + 10⁻³ | `C0/upper.py` | exact dyadic |
| Remark 5.6 (a certified gap) | σ_ess ∩ (1.89 e^{c_R}, 2.11 e^{c_R}) = ∅ for the period-8 prime set: stored fibre certificates at 129 Bloch nodes and the operator error budget | `spectral-gap/floquet_gap.py` | exact rational |
| Prop 9.14(1) | tail block ‖A′‖ ≤ 0.27085689 | `explicit-bounds/tail_block.py 2 2000` | Arb |
| Prop 9.14(2) | primes ≤ 10²¹⁶: κ² ≤ 9.86343 < π² | `explicit-bounds/primes_below_1e216.py 890` | Arb |
| Prop 9.14(3) | C_d ≤ 1.19914, 1.58582, 1.72142 (d = 1, 2, 3) | `explicit-bounds/first_d_primes.py 1e16 1 2 3` | Arb |
| Prop 9.14(3) | C_d ≤ 1.80494, 1.84758, 1.88301, 1.90873 (d = 4, …, 7) | `explicit-bounds/first_d_primes.py 1e10 4 5 6 7` | Arb |
| Prop 9.14(3) | 1.4578430 ≤ π*₂ ≤ 1.4579223 | `explicit-bounds/pi_star_2.py` | Arb |
| Prop 9.14(3) | C₂ ≥ 1.54792 and ‖q‖₁ ≤ 1.0005030 for the witness in `c2_witness.json` | `explicit-bounds/c2_lower.py 8192` | Arb, with a priori IEEE rounding bounds (needs numpy) |
| Prop 4.24 (Positive Nehari fails: two-atom witness) | the phase integral exceeds 17/100 > 1/A, so D(σ) > C₂(σ) | `positive-nehari/two_atom_phase.py` | Arb, with outward-rounded binary64 intervals (needs numpy) |
| Thm 5.12, Cor 5.13 (no eigenvalues, R = {p ≡ 1 mod 4}) | P ∈ [0.856108, 0.856110], K ≤ 0.45751, c_R ∈ [0.579821, 0.579822], L_R ∈ [4.30012, 4.30014]; both scalar inequalities of the proof | `eigenvalues-mod4/zeta_mod4.py` | Arb |
| Lemma 6.20 (Thm 6.19, unimodality) | N(19/10) < 1.2313, N(13/5) < 1.3438 (Schur tests, 2291 cells each), N(3) > 1.6031 (Rayleigh quotient), Λ_{8/5} > 1.5169; the slopes of Lemma 6.21 | `bergman-unimodality/unimodality.py` | Arb, with directed binary64 sums (needs numpy) |
| Lemma 6.22 (Thm 6.19) | Φ_q''(α) > 0 for α ∈ [8/5, 3], 10⁻⁴⁰ ≤ q ≤ 8, on the boxes of the stored tree `convex_tree.json` (regenerate with `make_convex_tree.py`) | `bergman-unimodality/unimodality.py` | Arb |
| Prop 8.10 (clipped Gaussian moment) | Φ/(2^{n−1}K₋³) ≤ −0.01165 for 194 ≤ n ≤ 3998 on the 11 blocks in `clipped_blocks.json` (regenerate with `make_blocks.py`); constants of Prop 8.9, N = 3999 | `clipped-gaussian/clipped_finite.py` | Arb |
| Thm 9.15 (three primes) | C₃ > 1.5990217 > 1.599; ‖T‖ ≤ 1.56179569435114 by Schur's test with the weights in `schur_weights.json` (regenerate with `make_weights.py`), ‖E‖ < 0.000529, π*₃ < 1.56232458 < 1.5624 | `three-prime/pi3_upper.py` | Arb, with correctly rounded binary64 sums (needs numpy) |
| Thm 8.6 (generator inequality); Thm 9.2 (column estimate) | the displayed polynomial identities and their signs | `symbolic/symbolic_checks.py` | exact (sympy) |

## Regenerating the spectral-gap certificate

`run_all.py` checks the stored fibre certificates in `spectral-gap/floquet_gap.json`.
To regenerate them from scratch (Arb entry enclosures, exact dyadic LDL
inertia and residual bounds; needs python-flint, numpy and scipy):

```
cd spectral-gap
python3 regenerate.py --nodes all --compare
```

This takes about 40 s per node, 129 nodes; `--nodes` also accepts a
comma-separated list, so the run can be split over several processes.
`--compare` also checks that each regenerated row equals the stored one; that
needs binary64 results identical to the original run, while the pass/fail
verdict does not.

## The Gaussian probabilistic certificate (Remark 11.10)

`gaussian-certificate/` contains the scripts and data for the statement that
the Gaussian constant is at least 3.1450, except on an event of probability at
most 2·10⁻³⁰. **This is not a proof.** It assumes that numpy's PCG64 generator
behaves like independent draws and that floating-point arithmetic obeys the
IEEE-754 rounding model. The run used seed 20261008, which is the run date.

```
cd gaussian-certificate
python rigor.py                              # exact constants -> rigor.json, Ld.npy, ell.npy (about 10 s)
python sample.py 20261008 5000               # 1e8 samples -> samples_20261008_5000.json (about 40 min, 16 cores)
python certify.py samples_20261008_5000.json 30   # -> certificate.json
python crosscheck.py 20261008                # optional extended-precision check
```

`candidate.json` holds the fixed quadratic as exact binary64 data. It was
produced by `freeze_candidate.py` from optimizer output that is not part of
this folder. `run_20261008.log` and `certificate.json` are the certified run.
numpy's `Generator` streams are not guaranteed stable across numpy versions,
so an exact rerun needs the numpy version of the original run (2.x, October
2026).
