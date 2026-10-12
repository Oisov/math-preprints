"""Numerical tools for "Zeros of optimal polynomial approximants in H^p".

Two kinds of computations appear in the paper:

* the limit problem of Sections 2-4: the convex function
  Psi_phi(v) = int_T |1-w|^(p-2) |R_phi(w) + v phi(w)|^p dm(w)
  and the boundary integral (5) deciding the sign of its derivative;
* optimal polynomial approximants (OPAs) of actual targets, computed by a
  damped Newton method on a quadrature rule adapted to the target.

Only numpy and scipy are used. Everything here is numerical evidence; the
proofs in the paper do not depend on it, and the exact certificate of
Section 7.2 lives in certificate.py.
"""

import warnings

import numpy as np
from numpy.polynomial import polynomial as P
from scipy import integrate, optimize


# ---------------------------------------------------------------------------
# Integrals over the circle with the weight |1-w|^gamma
# ---------------------------------------------------------------------------

def weighted_mean(fun, gamma, tol=1e-12):
    """Return int_T |1-w|^gamma fun(w) dm(w) for fun with fun(conj w) = conj fun(w).

    Such integrals are real and equal (1/pi) int_0^pi (2 sin(t/2))^gamma
    Re fun(e^{it}) dt. The factor t^gamma is handed to QUADPACK as an
    algebraic endpoint weight, so the integrable singularity at w=1 for
    gamma<0 costs nothing.
    """
    def g(t):
        smooth = np.sinc(t / (2 * np.pi)) ** gamma  # (2 sin(t/2)/t)^gamma
        return smooth * np.real(fun(np.exp(1j * t)))

    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", integrate.IntegrationWarning)
        val, _ = integrate.quad(g, 0.0, np.pi, weight="alg", wvar=(gamma, 0.0),
                                epsabs=tol, epsrel=tol, limit=500)
    return val / np.pi


# ---------------------------------------------------------------------------
# Profiles and the limit problem (Sections 2-4)
# ---------------------------------------------------------------------------

class Profile:
    """A polynomial phi with real coefficients and phi(1) = 1.

    Coefficients are in ascending order: phi(w) = sum_k c[k] w^k.
    """

    def __init__(self, coeffs, name=None):
        self.c = np.asarray(coeffs, dtype=float)
        if abs(self.c.sum() - 1) > 1e-12:
            raise ValueError("a profile must satisfy phi(1) = 1")
        one_minus_phi = -self.c.copy()
        one_minus_phi[0] += 1
        self.r, rem = P.polydiv(one_minus_phi, [1.0, -1.0])  # R = (1-phi)/(1-w)
        assert np.allclose(rem, 0)
        self.name = name or " ".join(f"{a:+g}w^{k}" for k, a in enumerate(self.c))

    @classmethod
    def family(cls, lam, p):
        """phi_lambda(w) = 1 + lambda (1-w)(1+sigma w), sigma = sign(2-p)."""
        s = 1.0 if p < 2 else -1.0
        return cls([1 + lam, lam * (s - 1), -lam * s],
                   name=f"phi_lambda, lambda={lam:g}")

    def phi(self, w):
        return P.polyval(w, self.c)

    def R(self, w):
        return P.polyval(w, self.r)

    def min_modulus(self):
        """Smallest modulus of a zero of phi (phi must not vanish in the closed disc)."""
        roots = P.polyroots(self.c)
        return np.min(np.abs(roots)) if len(roots) else np.inf

    # --- the limit problem -------------------------------------------------
    def psi(self, v, p):
        return weighted_mean(lambda w: np.abs(self.R(w) + v * self.phi(w)) ** p, p - 2)

    def dpsi(self, v, p):
        """Real derivative of Psi at real v, formula (1)."""
        def integrand(w):
            L = self.R(w) + v * self.phi(w)
            return np.abs(L) ** (p - 2) * np.real(np.conj(L) * self.phi(w))
        return p * weighted_mean(integrand, p - 2)

    def criterion(self, p):
        """Psi'(1/2) by the boundary identity (5): p int |1-w|^g |L|^g Re h."""
        def integrand(w):
            L = self.R(w) + 0.5 * self.phi(w)
            h = 0.5 + w * self.R(w)
            return np.abs(L) ** (p - 2) * np.real(h)
        return p * weighted_mean(integrand, p - 2)

    def vstar(self, p):
        """The minimizer of Psi over the real line (it is real by Lemma 1)."""
        res = optimize.minimize_scalar(lambda v: self.psi(v, p), bracket=(0.0, 1.0),
                                       options={"xtol": 1e-12})
        return res.x


def scaled_criterion(eps, p):
    """Right-hand side of (6): the sign of Psi'(1/2) for phi_lambda, eps = 1/lambda."""
    s = 1.0 if p < 2 else -1.0

    def integrand(w):
        G = 1 + s * w
        Pw = (1 + w) * G
        return np.abs(Pw - eps) ** (p - 2) * (eps / 2 - np.real(w * G))
    return weighted_mean(integrand, p - 2)


def lambda_threshold(p, eps_min=1e-4):
    """lambda_0(p): the smallest lambda beyond which Psi'(1/2) < 0 for phi_lambda.

    The scaled criterion is evaluated on a logarithmic grid in eps = 1/lambda;
    every sign change found is reported, then refined with Brent's method.
    For p < 2 the scan stops at eps = 3.9 (lambda > 0.256), away from eps = 4
    where P - eps vanishes at w = 1.
    """
    eps = np.geomspace(3.9 if p < 2 else 50.0, eps_min, 200)
    vals = np.array([scaled_criterion(e, p) for e in eps])
    changes = np.nonzero(np.sign(vals[:-1]) != np.sign(vals[1:]))[0]
    roots = [optimize.brentq(scaled_criterion, eps[i + 1], eps[i], args=(p,), xtol=1e-14)
             for i in changes]
    return [1 / r for r in roots]


def F_sigma_closed_form(p):
    """F_sigma of Lemma 6 via the beta function."""
    from scipy.special import beta
    g = p - 2
    s = 1.0 if p < 2 else -1.0
    i_gamma = beta((g + 1) / 2, (2 * g + 3) / 2) / np.pi  # I(gamma, 2 gamma + 2)
    return -s * g * 2 ** (3 * g + 1) / (3 * g + 4) * i_gamma


def F_sigma_quadrature(p):
    s = 1.0 if p < 2 else -1.0

    def integrand(w):
        G = 1 + s * w
        return np.abs((1 + w) * G) ** (p - 2) * np.real(w * G)
    return weighted_mean(integrand, p - 2)


# ---------------------------------------------------------------------------
# Quadrature on the circle for targets with a near-singularity at z = -1
# ---------------------------------------------------------------------------

def circle_rule(delta=None, panel_nodes=24, ratio=1.6, uniform=2048):
    """Nodes z_k on T and weights w_k with sum w_k h(z_k) ~ int_T h dm.

    With delta=None this is the uniform (trapezoidal) rule. Otherwise the
    rule is graded towards z = -1 at the scale delta: Gauss-Legendre panels
    whose lengths grow geometrically in the distance |t - pi| from delta/50 to
    pi. This resolves targets such as f_alpha, which vary on the scale
    delta = 1-alpha near -1 and are smooth elsewhere, with about a thousand
    nodes instead of the ~10^2/delta of a uniform grid.
    """
    if delta is None:
        t = 2 * np.pi * np.arange(uniform) / uniform
        return np.exp(1j * t), np.full(uniform, 1.0 / uniform)
    x, w = np.polynomial.legendre.leggauss(panel_nodes)
    n_panels = int(np.ceil(np.log(np.pi / (delta / 50)) / np.log(ratio)))
    edges = np.concatenate([[0.0], np.geomspace(delta / 50, np.pi, n_panels + 1)])
    s, ws = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        s.append((a + b) / 2 + (b - a) / 2 * x)
        ws.append((b - a) / 2 * w)
    s, ws = np.concatenate(s), np.concatenate(ws)
    t = np.concatenate([np.pi - s, np.pi + s])  # both sides of t = pi
    return np.exp(1j * t), np.concatenate([ws, ws]) / (2 * np.pi)


# ---------------------------------------------------------------------------
# Optimal polynomial approximants
# ---------------------------------------------------------------------------

def opa(fz, z, wts, n, p, tol=1e-14, maxit=200):
    """Coefficients (ascending) of the degree-n OPA to 1/f in H^p.

    fz are the values of f at the quadrature nodes z with weights wts. The
    objective sum_k wts_k |1 - q(z_k) f(z_k)|^p is smooth and strictly convex
    in the 2(n+1) real coefficients; its gradient and Hessian are explicit,
    so a damped Newton method started from the H^2 solution converges in a
    handful of steps.
    """
    A = fz[:, None] * z[:, None] ** np.arange(n + 1)
    sw = np.sqrt(wts)
    c0 = np.linalg.lstsq(sw[:, None] * A, sw, rcond=None)[0]  # p = 2 start
    M = np.hstack([A, 1j * A])  # e = 1 - M x for real x = (Re c, Im c)
    x = np.concatenate([c0.real, c0.imag])

    def objective(x):
        return wts @ np.abs(1 - M @ x) ** p

    f_old = objective(x)
    for _ in range(maxit):
        e = 1 - M @ x
        a = np.maximum(np.abs(e), 1e-300)
        W = wts * a ** (p - 2)
        grad = -p * np.real((W * np.conj(e)) @ M)
        Rr = np.real((np.conj(e) / a)[:, None] * M)
        H = p * (np.real(M.conj().T @ (W[:, None] * M)) + (p - 2) * Rr.T @ (W[:, None] * Rr))
        step = -np.linalg.solve(H, grad)
        decrement = -grad @ step
        if decrement < tol * max(f_old, 1e-300):
            break
        t = 1.0
        while objective(x + t * step) > f_old - 0.25 * t * decrement and t > 1e-12:
            t /= 2
        x = x + t * step
        f_old = objective(x)
    c = x[: n + 1] + 1j * x[n + 1:]
    return c, f_old ** (1 / p)


def zeros(c):
    """Zeros of the polynomial with ascending coefficients c."""
    c = np.trim_zeros(np.asarray(c), "b")
    return P.polyroots(c) if len(c) > 1 else np.array([])


def f_alpha(profile, alpha):
    """The rational target f_alpha = phi(J_alpha)/(1 + alpha z) of Section 2."""
    def f(z):
        J = (z + alpha) / (1 + alpha * z)
        return profile.phi(J) / (1 + alpha * z)
    return f


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def heading(title, where):
    print(f"\n{title}\n{'-' * len(title)}  [{where}]")


def table(headers, rows):
    cells = [[str(h) for h in headers]] + [[str(x) for x in r] for r in rows]
    widths = [max(len(r[i]) for r in cells) for i in range(len(headers))]
    for k, r in enumerate(cells):
        print("  " + "  ".join(x.rjust(w) for x, w in zip(r, widths)))
        if k == 0:
            print("  " + "  ".join("-" * w for w in widths))


def check(label, ok):
    print(f"  {'ok ' if ok else 'FAIL'}  {label}")
    return ok
