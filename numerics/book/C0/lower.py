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
    gap = I(hilbert) + regular - pi*norm - I(error*norm)
    assert gap.lo > 0, 'No strict excess certified'
    assert F(gap.lo,S)/norm > F(6,10**8)
    result = {
        'status': 'certified',
        'precision_bits':192, 'degree':degree,
        'coefficient_denominator':10**12, 'coefficient_numerators':numerators,
        'norm_squared':str(norm), 'singular_form':str(hilbert),
        'regular_form_interval':regular.rational_bounds(),
        'pi_interval':pi.rational_bounds(), 'uniform_kernel_error':str(error),
        'gap_above_2pi_times_norm_interval':gap.rational_bounds(),
        'certified_lower_excess_fraction':str(F(gap.lo,S)/norm),
    }
    out = Path(__file__).with_name('lower_output.json')
    out.write_text(json.dumps(result, indent=2)+'\n')
    print('Exact interval sign: positive')
    print('Lower excess above 2pi:', float(F(gap.lo,S)/norm))
    print('Kernel error bound:', float(error))
    print('Arithmetic interval width:', float(F(gap.hi-gap.lo,S)))


if __name__ == '__main__':
    main()

