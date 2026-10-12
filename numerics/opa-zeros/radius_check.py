"""Numerical sanity checks for Theorem 5 of the paper (rho_1(4) = sqrt(7/8)).

None of the proofs depend on this script. It checks, in floating point:
  1. the Mobius form of the optimality identity on an actual H^4 OPA with a zero in D;
  2. the pointwise identity  Re(g s) = sigma (u - Re(T h)/(2 sigma))^2 + D/4 + Re(s conj h^2);
  3. the integral identity   int D = int omega y^2 + Re H(w)^2  for random H with H(0)=0;
  4. the closed forms of int 1/omega, int P/omega, int P^2/omega and m(w) < 1;
  5. Re Pi_H(w) > 0 for random H at w = 0.935 < sqrt(7/8);
  6. the two-pole optimum: max |arg W_4(C)| = arctan(1/sqrt 7), and the p -> 2 constant.
Requires numpy and scipy.
"""
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize

rng = np.random.default_rng(0)
ok = True


def check(label, cond):
    global ok
    ok &= bool(cond)
    print(("ok   " if cond else "FAIL ") + label)


# 1. OPA identity -------------------------------------------------------------
gc = [10, -15, 22, -29, 36, -41, 44, -45, 44, -41, 36, -30, 23, -17, 12, -7, 4]
M = 2**12
z = np.exp(2j * np.pi * np.arange(M) / M)
g = np.polyval(gc[::-1], z)
phi = np.stack([g, 1j * g, z * g, 1j * z * g])
x = np.zeros(4)
for _ in range(100):  # Newton for min mean |1 - (A+Bz) g|^4 (exact for this grid size)
    E = 1 - x @ phi
    R = np.real(np.conj(E) * phi)
    grad = -4 * np.mean(np.abs(E)**2 * R, 1)
    Hs = 8 * R @ R.T / M + 4 * np.real((np.abs(E)**2 * phi.conj()) @ phi.T) / M
    x = x - np.linalg.solve(Hs, grad)
A, B = x[0] + 1j * x[1], x[2] + 1j * x[3]
a = -A / B
print("OPA zero a =", a)
h = (A + B * z) * g
K = np.abs(1 - h)**2 * (1 - h)
I1 = np.mean((1 - K) * np.conj(h) / (1 - np.conj(a) * z))
check(f"identity int (1-K) conj(h)/(1-conj(a) z) dm = 0  (|I|={abs(I1):.1e})", abs(I1) < 1e-10)
# Mobius form: rotate so that a = r > 0, H(xi) = h(e^{i al} (xi+r)/(1+r xi)), Pi_H(-r) = 0
r, al = abs(a), np.angle(a)
xi = np.exp(2j * np.pi * np.arange(2**20) / 2**20)  # fine grid: the map is very distorting
zz = np.exp(1j * al) * (xi + r) / (1 + r * xi)
H = np.polyval(gc[::-1], zz) * (A + B * zz)
G = (1 - np.abs(1 - H)**2 * (1 - H)) * np.conj(H)
Pi = np.mean(G / (1 + r * np.conj(xi)))
H0 = np.polyval(gc[::-1], a) * (A + B * a)
check(f"Mobius form: H(0)={abs(H0):.1e}, Pi_H(-|a|) = 0 (|Pi|={abs(Pi):.1e})",
      abs(Pi) < 1e-9 and abs(H0) < 1e-12)

# 2. pointwise identity ---------------------------------------------------------
for _ in range(5):
    w = rng.uniform(-0.93, 0.93)
    th = rng.uniform(0, 2 * np.pi)
    s = 1 / (1 - w * np.exp(-1j * th))
    hh = rng.normal() + 1j * rng.normal()
    sig, tau = s.real, s.imag
    u = abs(hh)**2
    gval = (1 - abs(1 - hh)**2 * (1 - hh)) * np.conj(hh)
    T = s + 2 * np.conj(s)
    D = 8 * sig * u - np.real(T * hh)**2 / sig
    rhs = sig * (u - np.real(T * hh) / (2 * sig))**2 + D / 4 + np.real(s * np.conj(hh)**2)
    check(f"pointwise identity (err {abs(np.real(gval * s) - rhs):.1e})",
          abs(np.real(gval * s) - rhs) < 1e-10)

# 3. integral identity ----------------------------------------------------------
M = 2**13
xi = np.exp(2j * np.pi * np.arange(M) / M)
for w in (0.3, 0.9, 0.935):
    c = (rng.normal(size=12) + 1j * rng.normal(size=12)) / np.arange(1, 13)
    H = np.polyval(np.r_[c[::-1], 0], xi)
    S = 1 / (1 - w * np.conj(xi))
    sig, tau = S.real, S.imag
    T = S + 2 * np.conj(S)
    D = 8 * sig * abs(H)**2 - np.real(T * H)**2 / sig
    om = 7 * sig - tau**2 / sig
    Hw = np.polyval(np.r_[c[::-1], 0], w)
    lhs, rhs = np.mean(D), np.mean(om * H.imag**2) + np.real(Hw**2)
    check(f"w={w}: int D = int omega y^2 + Re H(w)^2  ({lhs:.10f} vs {rhs:.10f})",
          abs(lhs - rhs) < 1e-9 * (1 + abs(lhs)))

# 4. closed forms ---------------------------------------------------------------
for w in (0.2, 0.6, 0.9, 0.93, 0.935):
    q = np.sqrt(7 - 8 * w * w)

    def S_(t): return 1 / (1 - w * np.exp(-1j * t))

    def om_(t):
        s = S_(t)
        return (7 * s.real**2 - s.imag**2) / s.real

    def P_(t): return (1 - w * w) / (1 - 2 * w * np.cos(t) + w * w)

    def I(f): return quad(f, 0, np.pi, limit=800, epsabs=1e-13, epsrel=1e-12)[0] / np.pi

    A0, A1, A2 = I(lambda t: 1 / om_(t)), I(lambda t: P_(t) / om_(t)), I(lambda t: P_(t)**2 / om_(t))
    a_cf = (1 + 2 * np.sqrt(7) * q - q * q) / (8 * np.sqrt(7) * q)
    b_cf = (1 + q * q) / (8 * np.sqrt(7) * q)
    m = a_cf - b_cf**2 / a_cf
    good = max(abs(A0 - a_cf), abs(A2 - a_cf), abs(A1 - b_cf)) < 1e-8 * a_cf
    check(f"w={w}: closed forms (a={a_cf:.6f}, b={b_cf:.6f}), m(w)={m:.6f} < 1", good and m < 1)

# 5. Re Pi > 0 near the threshold -------------------------------------------------
w = 0.935
S = 1 / (1 - w * np.conj(xi))
worst = np.inf
for _ in range(2000):
    N = rng.integers(1, 40)
    c = (rng.normal(size=N) + 1j * rng.normal(size=N)) * rng.uniform(0.05, 3) / np.sqrt(N)
    H = np.polyval(np.r_[c[::-1], 0], xi)
    G = (1 - np.abs(1 - H)**2 * (1 - H)) * np.conj(H)
    worst = min(worst, np.real(np.mean(G * S)) / np.mean(np.abs(G * S)))
check(f"Re Pi_H(0.935) > 0 on 2000 random H (min normalized {worst:.3f})", worst > 0)

# 6. two-pole optimum -----------------------------------------------------------
Ms = 2**16
sgrid = 2 * np.pi * (np.arange(Ms) + 0.5) / Ms
Z = (1 + np.exp(1j * sgrid)) / 2  # H(t) on the circle; |H|^2 dt/pi = ds/2pi


def W(p, C):
    hh = C * Z
    E = 1 - hh
    return np.mean((1 - np.abs(E)**(p - 2) * E) / hh)


C0 = (3 - 1j * np.sqrt(7)) / 2
check("W_4((3-i sqrt7)/2) = (7 - i sqrt7)/4", abs(W(4, C0) - (7 - 1j * np.sqrt(7)) / 4) < 1e-12)
r_ = minimize(lambda v: -abs(np.angle(W(4, v[0] + 1j * v[1]))), [2, -1], method="Nelder-Mead",
              options={"xatol": 1e-10, "fatol": 1e-14})
check("max |arg W_4| = arctan(1/sqrt 7)", abs(-r_.fun - np.arctan(1 / np.sqrt(7))) < 1e-9)


def Phi(C):
    hh = C * Z
    return np.mean((1 - hh) * np.log(np.abs(1 - hh)) / hh)


r_ = minimize(lambda v: -abs(Phi(v[0] + 1j * v[1]).imag), [1.6, -2.2], method="Nelder-Mead",
              options={"xatol": 1e-12, "fatol": 1e-15})
print("p->2 two-pole constant sup (Im Phi)^2/2 =", r_.fun**2 / 2, "at C =", r_.x)
print("ALL OK" if ok else "SOME CHECK FAILED")
