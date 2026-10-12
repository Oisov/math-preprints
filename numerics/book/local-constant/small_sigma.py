"""Theorem 2.8 (local constant), analytic part sigma <= 1/100.

The margin g(s) decreases in s; the theorem needs g(1/100) > 0 (about 0.0926).
Requires python-flint (Arb ball arithmetic).
"""
# analytic small-sigma part: need g(s)=4 d^{-1/4}(1+s/d)^{-1/2} - I_Phi - 4 d^{1/8} E(d) e^{beta d} s^{5/8} >= 0
from flint import arb, ctx
ctx.prec=64
d=arb(2); beta=arb(3)/10
def E(t): return (1+7*t/6+t*t/3).sqrt().sqrt()
def M(t): return (1+7*t/12+t*t/12).sqrt()
def Phi(t): return E(t)*M(t)*(beta*t).exp()
# I_Phi upper Riemann sum; Phi increasing
t1=arb('1e-6')
x=(arb(7)/12+beta)*t1+t1*t1/8
I=x.exp()*((arb(7)/12+beta)*arb(4)/3*t1**(arb(3)/4)+arb(1)/8*arb(4)/7*t1**(arb(7)/4))
t=1e-6; r=1.001
while t<2:
    tn=min(t*r,2.0)
    I+= (Phi(arb(tn))-1)*4*(arb(t)**(-arb(1)/4)-arb(tn)**(-arb(1)/4))
    t=tn
print("I_Phi <=",I.upper())
for s in ['1e-4','3e-4','1e-3','2e-3','3e-3','5e-3','1e-2']:
    s=arb(s)
    g=4*d**(-arb(1)/4)*(1+s/d)**(-arb(1)/2) - I - 4*d**(arb(1)/8)*E(d)*(beta*d).exp()*s**(arb(5)/8)
    print(s, g.lower())

assert g.lower()>0, "margin at s=1/100 is not positive"
print("PASS: margin at sigma=1/100 is positive")
