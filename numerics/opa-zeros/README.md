# Numerics for "Zeros of optimal polynomial approximants in H^p"

Companion code for the paper. None of the proofs depend on it. One script is an
exact certificate; the others reproduce every number quoted in the paper and
check the main lemmas on examples.

```sh
python3 run_all.py        # everything, about 35 seconds
python3 certificate.py    # the exact certificate alone (standard library only)
```

Requires Python 3 with numpy and scipy; `certificate.py` needs neither.
Each script prints its tables and ends with `ok`/`FAIL` lines, and exits nonzero
if a check fails.

| Script | Paper | What it does |
|---|---|---|
| `certificate.py` | §7.2 | Exact rational proof that the degree-16 integer polynomial `g` has no zeros in the closed disc (Schur–Cohn) and that its degree-one OPA in H⁴ vanishes in 𝔻 (convexity and tangent bounds). Also certifies the degree-14 variant. |
| `check_limit.py` | Lemmas 1–7, Corollary 2, §7.1, §7.3 | The limit problem Ψ_φ: the beta-integral formula for F_σ, the boundary identity (formulas (1) and (5) agree), the value 1 at p = 2, the p = 4 cubic, the p < 4/3 profile 2 − w, the thresholds λ₀(p) quoted in §7.3, and the limit of (6). |
| `check_opas.py` | Lemma 2, Theorem 1, Corollary 1, §7.5 | Actual OPAs. The zero of the OPA to 1/f_α enters 𝔻 as α → 1, at depth ≈ (2v* − 1)(1 − α). Taylor polynomials of f_α give zero-free polynomial targets with the same property. g(zⁿ) has OPA A + Bzⁿ. The deep p = 4 example has \|zero\| = 0.9968. |
| `check_bound.py` | Lemma 11, Theorem 3 | The cone lemma on a grid of 4·10⁶ points, the identity (10) on actual OPAs, and 4,500 OPA zeros of random polynomial targets, all outside radius 2√(p−1)/p. |
| `opa.py` | | The shared library (see below). |
| `sturm_p4.py`, `radius_check.py` | Theorem 5 | Exact Sturm and rational checks for the p = 4 radius √(7/8): the square completion, the closing identity, and actual OPAs approaching the radius. |
| `exact_p6_verify.py` | Proposition 2 | Exact rational verification (standard library) of the sum-of-squares certificates: `python3 exact_p6_verify.py exact_p6_data_0p8607` (p = 6, ρ₁(6) > 0.8607) and `python3 exact_p6_verify.py exact_p8_data_0p799` (p = 8, ρ₁(8) > 0.799). The data modules hold the rational Gram matrices. |
| `exact_profiles.py` | §9 (depth remark) | Exact rational values of B₆ and B₈ for the polynomial profiles behind ρ₁(6) ≤ 0.860918 and ρ₁(8) ≤ 0.800639. |
| `exact_p6_search.py`, `exact_p6_round.py` | Proposition 2 | How the certificates were found (semidefinite programming with cvxpy, then rounding). Not needed to check them. |
| `sos_*.py`, `radius_general_p.py`, `evenp_*.py` | §9 | Exploratory numerics for general even p (local SOS bounds, profile optimization, duality). Not proofs. |

## Methods

- **Weighted circle integrals.** Integrals of the form ∫ |1 − w|^γ F(w) dm(w)
  are computed with QUADPACK. The factor θ^γ is passed as an algebraic
  endpoint weight, so the singularity at w = 1 for p < 2 costs nothing, and
  the results are accurate to about 1e-12.
- **OPAs.** The OPA minimizes a smooth, strictly convex function of the
  2(n+1) real coefficients, Σ_k w_k |1 − q(z_k) f(z_k)|^p. Its gradient and
  Hessian are explicit, so a damped Newton method started from the H² solution
  converges in a few steps, in milliseconds.
- **Graded quadrature.** f_α varies on the scale δ = 1 − α near z = −1 and is
  smooth elsewhere. The quadrature rule uses Gauss–Legendre panels that grow
  geometrically away from −1, which needs about a thousand nodes for any δ down
  to 10⁻⁵. A uniform grid would need about 100/δ nodes. Two different graded
  rules agree to about 1e-9 relative to δ.

## Observations

- The depth (1 − |zero|)/(1 − α) converges to 2v* − 1 slowly when p is close
  to 1. For p = 1.2 the correction decays roughly like (1 − α)^0.2, but Aitken
  extrapolation of the last three values recovers the limit (0.353 against
  0.349).
- Random polynomial targets never produced a zero in 𝔻: 0 of 4,500 zeros at
  p ∈ {1.2, 1.5, 3, 4, 6}. Interior zeros need targets built for the purpose,
  such as those in the paper.
