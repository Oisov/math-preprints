"""Run every check for "Zeros of optimal polynomial approximants in H^p".

Run: python3 run_all.py      (about 15 seconds; needs numpy and scipy)
"""

import time

import certificate
import check_bound
import check_limit
import check_opas
import sturm_p4
import subprocess, sys, os

PARTS = [("Exact certificate, Section 9.2", certificate),
         ("Limit problem, Sections 2-4 and 9.1-9.3", check_limit),
         ("OPAs of actual targets, Theorem 1, Corollary 1, Section 9.5", check_opas),
         ("Lower bound, Theorem 3", check_bound),
         ("Theorem 4 (Section 7) and Section 9.6", sturm_p4)]


class _Radius:
    """numerics/radius_check.py is a script; run it and report its status."""
    @staticmethod
    def main():
        here = os.path.dirname(os.path.abspath(__file__))
        out = subprocess.run([sys.executable, os.path.join(here, "radius_check.py")],
                             capture_output=True, text=True)
        print(out.stdout[-2000:])
        return out.returncode == 0 and "ALL OK" in out.stdout


PARTS.append(("Theorem 5, the exact radius at p=4 (Section 8)", _Radius))


class _ExactP6:
    """Exact rational verification of Proposition 2 (rho_1(6) > 0.8607)."""
    @staticmethod
    def main():
        import exact_p6_verify
        sys.argv = ["exact_p6_verify.py", "exact_p6_data_0p8607"]
        return exact_p6_verify.main()


PARTS.append(("Proposition 2, rho_1(6) > 0.8607 (exact certificate)", _ExactP6))


class _ExactP8:
    """Exact rational verification of Proposition 2 (rho_1(8) > 0.799)."""
    @staticmethod
    def main():
        import exact_p6_verify
        sys.argv = ["exact_p6_verify.py", "exact_p8_data_0p799"]
        return exact_p6_verify.main()


PARTS.append(("Proposition 2, rho_1(8) > 0.799 (exact certificate)", _ExactP8))


class _Profiles:
    """Exact B_p(G) for the profiles giving rho_1(6) <= 0.860918 and rho_1(8) <= 0.800639."""
    @staticmethod
    def main():
        import exact_profiles
        return exact_profiles.main()


PARTS.append(("Section 9.6, profile upper bounds at p=6, 8 (exact)", _Profiles))


def main():
    results = []
    for title, module in PARTS:
        print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")
        start = time.perf_counter()
        ok = module.main()
        results.append((title, ok, time.perf_counter() - start))
    print(f"\n{'=' * 78}\nSummary\n{'=' * 78}")
    for title, ok, sec in results:
        print(f"  {'ok ' if ok else 'FAIL'}  {sec:5.1f} s  {title}")
    return all(ok for _, ok, _ in results)


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
