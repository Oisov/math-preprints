"""Directed dyadic arithmetic certificate for a fixed C0 polynomial witness.

Only Python's standard library is used. Floating point is used for printed
summaries, never for the sign decision. See Proposition 3.8 (constant at p=2) of the book.
"""
from fractions import Fraction as F
from math import factorial, comb
from pathlib import Path
import json

S = 1 << 192


class I:
    def __init__(self, x=0, hi=None):
        if hi is not None:
            self.lo, self.hi = x, hi
        else:
            x = F(x)
            self.lo = x.numerator * S // x.denominator
            self.hi = -((-x.numerator * S) // x.denominator)

    def __add__(self, b):
        b = b if isinstance(b, I) else I(b)
        return I(self.lo + b.lo, self.hi + b.hi)

    __radd__ = __add__

    def __neg__(self):
        return I(-self.hi, -self.lo)

    def __sub__(self, b):
        return self + (-b if isinstance(b, I) else -F(b))

    def __mul__(self, b):
        b = b if isinstance(b, I) else I(b)
        p = [a * c for a in (self.lo, self.hi) for c in (b.lo, b.hi)]
        return I(min(p) // S, -((-max(p)) // S))

    __rmul__ = __mul__

    def rational_bounds(self):
        return [str(F(self.lo, S)), str(F(self.hi, S))]


def log_fraction(q, terms=70):
    x = (q - 1) / (q + 1)
    v = 2 * sum((x ** (2*j+1) / (2*j+1) for j in range(terms)), F(0))
    tail = 2*x**(2*terms+1) / ((2*terms+1)*(1-x*x))
    return I(v) + I(0, I(tail).hi)


def log_integer(n):
    k = n.bit_length() - 1
    return log_fraction(F(2)) * k + log_fraction(F(n, 2**k))


def atan_fraction(x, terms=160):
    v = sum(((-1)**j * x**(2*j+1)/(2*j+1) for j in range(terms)), F(0))
    next_term = (-1)**terms * x**(2*terms+1)/(2*terms+1)
    a, b = sorted((v, v+next_term))
    return I(I(a).lo, I(b).hi)


def exp_coefficients(logn, degree):
    result = [I(1)]
    for k in range(1, degree+1):
        result.append(result[-1] * (-logn) * F(1, k))
    return result


def cmul(a, b):
    return (a[0]*b[0]-a[1]*b[1], a[0]*b[1]+a[1]*b[0])


def main():
    degree = 96
    # Fixed exact witness, discovered numerically but now independent of that
    # search, its eigensolver, and all floating-point coefficient conversion.
    numerators = json.loads(Path(__file__).with_name('witness.json').read_text())['new_lower_witness']['coefficient_numerators']
    a = [(F(v, 10**12), F(0)) if n % 2 == 0 else (F(0), F(v, 10**12)) for n,v in enumerate(numerators)]
    norm = sum(((x*x+y*y)/F(2*n+1) for n,(x,y) in enumerate(a)), F(0))
    hilbert = F(0)
    for n in range(len(a)):
        for m in range(len(a)):
            if (n+m) % 2:
                prod = cmul((a[n][0], -a[n][1]), a[m])
                hilbert += prod[1] * F(2, (n-m)*(n+m+1))
    moments = []
    for k in range(degree+1):
        moments.append(tuple(sum((a[n][part]*F(factorial(k)**2, factorial(k-n)*factorial(k+n+1)) for n in range(min(k+1,len(a)))), F(0)) for part in range(2)))
    weights = []
    for k in range(degree+1):
        z = [F(0), F(0)]
        for l in range(k+1):
            p = cmul((moments[l][0], -moments[l][1]), moments[k-l])
            for j in range(2):
                z[j] += comb(k,l)*(-1)**(k-l)*p[j]
        z = cmul(tuple(z), [(F(1),F(0)), (F(0),F(1)), (F(-1),F(0)), (F(0),F(-1))][k%4])
        assert z[1] == 0
        weights.append(z[0])
    coefficients = [I(0) for _ in range(degree+1)]
    for n in range(1,64):
        for k, e in enumerate(exp_coefficients(log_integer(n), degree)):
            coefficients[k] = coefficients[k] + e*F(1,n)
    logs = log_integer(64)
    exp = exp_coefficients(logs, degree+1)
    for k in range(degree+1):
        coefficients[k] = coefficients[k] + exp[k+1] + exp[k]*F(1,128)
    bernoulli = [F(1,12), F(-1,720), F(1,30240), F(-1,1209600), F(1,47900160), F(-691,1307674368000)]
    assert logs.hi < 6*S
    assert sum((abs(b)*F(factorial(2*j+1),2)*F(64**2,64**(2*j))
                for j,b in enumerate(bernoulli,1)), F(0)) < 1
    for j, b in enumerate(bernoulli, 1):
        poly = [1]
        for l in range(1,2*j):
            new = [0]*(len(poly)+1)
            for k, v in enumerate(poly):
                new[k] += l*v
                new[k+1] += v
            poly = new
        for k in range(degree+1):
            c = sum((exp[k-l]*v for l,v in enumerate(poly) if l<=k), I(0))
            coefficients[k] = coefficients[k] + c*b*F(1,64**(2*j))
    regular = sum((c*w for c,w in zip(coefficients, weights)), I(0))
    pi = atan_fraction(F(1,5))*16 - atan_fraction(F(1,239))*4
    # EM on |w|<=2: |(1+w)_12|<=14!/2, Re(1+w)>=-1.
    em_error = F(691,1307674368000)*F(factorial(14),2)*F(1,10*64**10)
    # |E(w)|<32768 on |w|=2, so tail on |w|<=1 is <=2^-81.
    error = em_error + F(1,2**81)
    # Certified centered Taylor bounds for sup_{|t|<=1} |psi(1+it)|.
    from math import isqrt
    def abs_upper(x):
        return F(max(abs(x.lo), abs(x.hi)), S)
    def sqrt_upper(x):
        k = isqrt((x.numerator * S * S) // x.denominator)
        if F(k*k, S*S) < x:
            k += 1
        return F(k, S)
    bounds = []
    pieces = 128
    for jpiece in range(pieces):
        center = F(2*jpiece+1, 2*pieces)
        radius = F(1, 2*pieces)
        real_derivs = []
        imag_derivs = []
        for jderiv in range(degree+1):
            re = I(0)
            im = I(0)
            for k in range(jderiv, degree+1):
                term = coefficients[k] * (comb(k,jderiv) * center**(k-jderiv))
                if k % 2 == 0:
                    re = re + term * ((-1)**(k//2))
                else:
                    im = im + term * ((-1)**((k-1)//2))
            real_derivs.append(re)
            imag_derivs.append(im)
        d0sq = abs_upper(real_derivs[0])**2 + abs_upper(imag_derivs[0])**2
        b = sqrt_upper(d0sq)
        for jderiv in range(1, degree+1):
            b += (abs_upper(real_derivs[jderiv]) + abs_upper(imag_derivs[jderiv])) * radius**jderiv
        bounds.append(b)
    psi_sup_bound = max(bounds) + error
    # Rank-two baseline removes the constant and the leading imaginary linear kernel.
    c = F(57803, 100000)
    slope = F(728158454836767, 10**16)
    hs_squared = I(0)
    for k in range(degree+1):
        ck = coefficients[k] - c if k == 0 else coefficients[k] - slope if k == 1 else coefficients[k]
        for l in range(degree+1):
            total = k+l
            if total % 2 == 0:
                cl = coefficients[l] - c if l == 0 else coefficients[l] - slope if l == 1 else coefficients[l]
                factor = F(2, (total+1)*(total+2)) * (-1 if ((k-l)//2)%2 else 1)
                hs_squared = hs_squared + ck*cl*factor
    hs_poly_upper = sqrt_upper(F(hs_squared.hi, S))
    hs_remainder_upper = hs_poly_upper + error
    assert hs_remainder_upper < F(1,1000), hs_remainder_upper
    print('Rank-two remainder HS upper:', float(hs_remainder_upper))
    print('Rank-two remainder HS fraction:', hs_remainder_upper)
    pi_lower, pi_upper = F(pi.lo,S), F(pi.hi,S)
    assert pi_lower > F(31415,10000)
    assert pi_upper < F(11,3)
    delta = F(1,340000)
    # For q>0, scale into [1,2), then use the atanh series and its positive tail.
    def log_interval(q):
        k = 0
        while q >= 2:
            q /= 2
            k += 1
        while q < 1:
            q *= 2
            k -= 1
        return log_fraction(F(2))*k + log_fraction(q)
    def log_upper(q):
        return F(log_interval(q).hi, S)
    def log_lower(q):
        return F(log_interval(q).lo, S)
    m00_upper = log_upper((2*pi_upper+delta)/delta)/(2*pi_lower)
    L_upper = log_upper(2*pi_upper/delta)
    m22_upper = (L_upper**2 + 2*L_upper + 2 + L_upper**3/F(3) + 4)/(2*pi_lower)
    # K has one positive eigenvalue mu; its normalized eigenvector is alpha*e0+i*beta*e1.
    mu_upper = c + slope**2/(12*c)
    d2_upper = slope**2/(4*pi_lower**2*c**2)
    # The dilogarithm reflection identities give an explicit negative cross moment.
    # With M=log(1/h), the exact integral is <= -M^2/2+h; see the paired MD proof.
    h_upper = delta/(2*pi_lower)
    M_lower = log_lower(2*pi_lower/delta)
    assert log_upper(F(2)) < F(7,10)
    cross_magnitude_lower = (M_lower**2-2*h_upper)/(2*pi_upper)
    assert cross_magnitude_lower > 0
    alpha_d_lower = slope/(2*pi_upper*mu_upper*(1+slope**2/(12*c**2)))
    birman_schwinger_upper = mu_upper*(m00_upper+d2_upper*m22_upper-alpha_d_lower*cross_magnitude_lower)
    assert birman_schwinger_upper < F(999,1000), birman_schwinger_upper
    final_excess_upper = hs_remainder_upper + delta
    assert final_excess_upper < F(1,1028), final_excess_upper
    print('Rank-two Birman-Schwinger bound:', float(birman_schwinger_upper))
    print('Rank-two Birman-Schwinger fraction:', birman_schwinger_upper)
    print('Negative cross-moment magnitude lower bound:', float(cross_magnitude_lower))
    print('alpha*d lower bound:', float(alpha_d_lower))
    print('Rank-two model excess:', delta)
    print('Final C0 excess upper:', float(final_excess_upper))
    print('Final C0 excess exact fraction:', final_excess_upper)
    assert psi_sup_bound < F(58699,100000), psi_sup_bound
    print('Centered Taylor sup bound:', float(psi_sup_bound))
    print('Centered Taylor sup fraction:', psi_sup_bound)
    print('Centered Taylor partition:', pieces)
    gap = I(hilbert) + regular - pi*norm - I(error*norm)
    assert gap.lo > 0, 'No strict excess certified'
    assert F(gap.lo,S)/norm > F(6,10**8)
    result = {
        'status': 'rank_two_dilog_cross_independently_audited_2026-10-11T14:37Z',
        'precision_bits':192, 'degree':degree,
        'coefficient_denominator':10**12, 'coefficient_numerators':numerators,
        'norm_squared':str(norm), 'singular_form':str(hilbert),
        'regular_form_interval':regular.rational_bounds(),
        'pi_interval':pi.rational_bounds(), 'uniform_kernel_error':str(error),
        'gap_above_2pi_times_norm_interval':gap.rational_bounds(),
        'certified_lower_excess_fraction':str(F(gap.lo,S)/norm),
        'rank_two_global_upper_bound': {
            'status': 'independently_audited_2026-10-11T14:37Z',
            'model': 'S + c*1tensor1 + i*a*(t-u)',
            'constant_c': str(c), 'linear_coefficient_a': str(slope),
            'certified_hilbert_schmidt_remainder_upper': str(hs_remainder_upper),
            'model_spectral_excess_upper': str(delta),
            'birman_schwinger_scalar_upper': str(birman_schwinger_upper),
            'birman_schwinger_threshold': '999/1000',
            'negative_cross_magnitude_lower': str(cross_magnitude_lower),
            'logit_M_lower': str(M_lower),
            'h_upper': str(h_upper),
            'alpha_times_d_lower': str(alpha_d_lower),
            'm22_upper': str(m22_upper),
            'final_exact_excess_upper': str(final_excess_upper),
            'final_simple_excess_upper': '1/1028',
            'delta': str(delta)
        },
    }
    out = Path(__file__).with_name('upper_output.json')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print('Exact interval sign: positive')
    print('Lower excess above 2pi:', float(F(gap.lo,S)/norm))
    print('Kernel error bound:', float(error))
    print('Arithmetic interval width:', float(F(gap.hi-gap.lo,S)))


if __name__ == '__main__':
    main()


