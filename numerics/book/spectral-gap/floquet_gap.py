"""Validate the preserved period-8 certificate and its operator error budget.

Uses exact rational arithmetic and no third-party packages. This checks the
finite-matrix certificate stored in floquet_gap.json: inertia and residual
inequalities at 129 Bloch nodes and the operator error budget. It does not
regenerate the Arb entry enclosures or the dyadic LDL factors from scratch.
"""
from fractions import Fraction as F
from pathlib import Path
import json


def exp_bounds(x, terms=80):
    """Rational lower/upper bounds for exp(x), x >= 0."""
    if x < 0 or x >= terms + 2:
        raise ValueError("Taylor remainder assumptions fail")
    term = total = F(1)
    for n in range(1, terms + 1):
        term *= x / n
        total += term
    first_omitted = term * x / (terms + 1)
    return total, total + first_omitted / (1 - x / (terms + 2))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(data):
    expected = {
        "period": 8, "theta": "1/2", "matrix_size": 128,
        "image_cutoff": 12, "mesh_denominator": 128,
        "dyadic_bits": 44, "arb_bits": 128,
    }
    require(data["parameters"] == expected, "wrong operator parameters")
    require(
        data["source_sha256"] ==
        "15d589414e4b646e6e022815ebc48465610010a9a05962fd415e7f5ac2800223",
        "wrong source record",
    )
    rows = data["nodes"]
    require([r["node"] for r in rows] == list(range(129)),
            "missing, duplicated, or reordered Bloch nodes")
    for row in rows:
        require(F(row["max_entry_error_upper"]) >= 0,
                "negative entry-error bound")
        for threshold in (1, 3):
            cert = row[f"threshold_{threshold}"]
            require(cert["inertia"] == [1, 127],
                    f"wrong inertia at node {row['node']}, threshold {threshold}")
            alpha = F(cert["Linv2_times_total_residual_upper"])
            delta_sq = F(cert["D_min_singular_sq_lower"])
            # Recompute the inequality; do not rely on the recorded Boolean.
            require(alpha >= 0 and delta_sq > 0 and alpha**2 < delta_sq,
                    f"failed residual at node {row['node']}, threshold {threshold}")

    # The classical rational bounds 157/50 < pi < 22/7 are used below.
    pi_lower, pi_upper = F(157, 50), F(22, 7)
    require(pi_lower**2 / 4 > F(123, 50), "pi-square bound")
    exp246_lower, _ = exp_bounds(F(123, 50))
    require(exp246_lower > F(117, 10), "exponential lower bound")
    require(1 / exp246_lower < F(43, 500), "negative exponential bound")
    sinh_lower = (F(117, 10) - F(43, 500)) / 2
    sqrt_sinh_lower = F(2409, 1000)
    require(sqrt_sinh_lower**2 < sinh_lower, "square-root lower bound")
    exp123_lower, _ = exp_bounds(F(123, 100))
    require(exp123_lower > F(10, 3), "q < 3/10 bound")
    q = F(3, 10)
    psi_upper = 4 / (pi_lower * sqrt_sinh_lower)
    psi_upper += 4 * q**3 / (3 * pi_lower * (1 - q**2))
    derivative_upper = 1 / sqrt_sinh_lower + q**3 / (1 - q**2)
    require(psi_upper < F(55, 100), "modulation bound")
    require(derivative_upper < F(45, 100), "modulation derivative bound")
    fourier_tail_upper = F(1, 36) * q**21 / (1 - q**2)
    require(fourier_tail_upper < F(1, 10**12), "Arb Fourier-tail radius")

    _, exp055_upper = exp_bounds(F(55, 100))
    require(F(129, 16) * exp055_upper < 14, "midpoint row bound")
    require(12 * exp055_upper < 21, "Bloch derivative bound")
    _, exp_step_upper = exp_bounds(F(7, 160))
    exp24_lower, _ = exp_bounds(F(24))
    exp2_lower, _ = exp_bounds(F(2))
    midpoint = 14 * (exp_step_upper - 1)
    bloch = 21 * pi_upper / 256
    images = 16 * exp055_upper / (exp24_lower * (1 - 1 / exp2_lower))
    total = midpoint + bloch + images
    require(total < F(89, 100), "operator error budget >= 0.89")
    return len(rows), 2 * len(rows)


if __name__ == "__main__":
    path = Path(__file__).with_suffix(".json")
    nodes, inequalities = verify(json.loads(path.read_text()))
    print(f"PASS: {nodes} Bloch nodes; {inequalities} exact residual inequalities; "
          "inertia (1,127) at both thresholds; total operator error < 89/100.")
