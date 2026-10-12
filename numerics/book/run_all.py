"""Run the book's certificates and compare with the values printed in the book.

Usage: python3 run_all.py [--skip-arb]
The Arb scripts need python-flint and the symbolic checks need sympy; scripts
whose library is missing are skipped and reported. With --skip-arb only the
exact-arithmetic certificates run.
"""
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

# (description, script, args, needs_arb, check) -- check(stdout) -> bool
JOBS = [
    ("Prop 3.7(3): C0 > 2pi + 2.875e-7", "C0/lower.py", [], False,
     lambda o: "Exact interval sign: positive" in o
     and float(re.search(r"Lower excess above 2pi: (\S+)", o).group(1)) > 2.875e-7),
    ("Prop 3.7(3): C0 < 2pi + 1e-3", "C0/upper.py", [], False,
     lambda o: float(re.search(r"Final C0 excess upper: (\S+)", o).group(1)) < 1e-3),
    ("Remark 5.6: certified spectral gap", "spectral-gap/floquet_gap.py", [], False,
     lambda o: o.strip().splitlines()[-1].startswith("PASS")),
    ("Thm 2.8: local constant, sigma <= 1/100", "local-constant/small_sigma.py", [], True,
     lambda o: "PASS" in o),
    ("Thm 2.8: local constant, 268 cells", "local-constant/cells.py",
     ["0.01", "2", "1.02", "1.005"], True,
     lambda o: float(re.search(r"WORST \(\S+, (\S+)\)", o).group(1)) > 0.17),
    ("Prop 9.14(1): tail block <= 0.27085689", "explicit-bounds/tail_block.py",
     ["2", "2000"], True,
     lambda o: float(re.search(r"certified Schur bound (\S+)", o).group(1)) <= 0.27085689),
    ("Prop 9.14(2): kappa^2 <= 9.86343 < pi^2", "explicit-bounds/primes_below_1e216.py",
     ["890"], True,
     lambda o: float(re.search(r"kappa\^2<= (\S+)", o).group(1)) <= 9.86343),
    ("Prop 9.14(3): C_1, C_2, C_3", "explicit-bounds/first_d_primes.py",
     ["1e16", "1", "2", "3"], True,
     lambda o: o.count("C_d <=") == 3),
    ("Prop 9.14(3): C_4, ..., C_7", "explicit-bounds/first_d_primes.py",
     ["1e10", "4", "5", "6", "7"], True,
     lambda o: o.count("C_d <=") == 4),
    ("Prop 9.14(3): pi*_2", "explicit-bounds/pi_star_2.py", [], True,
     lambda o: "pi*_2 in" in o),
    ("Thm 8.6, Thm 9.2: symbolic identities (sympy)", "symbolic/symbolic_checks.py", [], "sympy",
     lambda o: "ALL OK" in o),
]


def main():
    skip_arb = "--skip-arb" in sys.argv
    if not skip_arb:
        try:
            import flint  # noqa: F401
        except ImportError:
            print("python-flint not found; running exact-arithmetic certificates only")
            skip_arb = True
    try:
        import sympy  # noqa: F401
        have_sympy = True
    except ImportError:
        print("sympy not found; skipping the symbolic checks")
        have_sympy = False
    ok = True
    for desc, script, args, arb, check in JOBS:
        if (arb is True and skip_arb) or (arb == "sympy" and not have_sympy):
            print(f"  skip   {desc}")
            continue
        t0 = time.time()
        p = subprocess.run([sys.executable, str(HERE / script), *args],
                           capture_output=True, text=True, cwd=HERE / Path(script).parent)
        dt = time.time() - t0
        try:
            good = p.returncode == 0 and check(p.stdout)
        except Exception:
            good = False
        ok &= good
        print(f"  {'ok  ' if good else 'FAIL'} {dt:7.1f} s  {desc}")
        if not good:
            print(p.stdout[-2000:], p.stderr[-2000:])
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
