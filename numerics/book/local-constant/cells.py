# Rigorous Schur-test verification for the prime-2-exact Hellinger kernel on (0,delta]
# K(s,t)=zeta(1+2s)^{1/4} zeta(1+2t)^{1/4} zeta(1+s+t)^{1/2} P(s,t),
# P^4=(1-a^2)(1-b^2)/(1-ab)^2, a=2^{-1/2-s}, b=2^{-1/2-t}; weight w(t)=t^{-1/2}e^{beta t}
from flint import arb, ctx
import sys, math
ctx.prec=64
DELTA=arb(2); BETA=arb(3)/10
B=arb(1)/4; B=B.gamma()**2/(2*arb.pi()).sqrt()
def ball(lo,hi):
    lo=arb(lo);hi=arb(hi); return (lo+hi)/2 + arb(0,1)*(hi-lo)/2  # crude: arb(mid,rad)
def mkball(lo,hi):
    m=(arb(lo)+arb(hi))/2; r=(arb(hi)-arb(lo))/2
    return arb(m.mid(), (r.upper()+m.rad()))
def P4(s,t):
    a2=arb(2)**(-1-2*s); b2=arb(2)**(-1-2*t); ab=arb(2)**(-1-s-t)
    return (1-a2)*(1-b2)/(1-ab)**2
def qrt(x): return x.sqrt().sqrt()
def row_upper(slo,shi,eps,ratio):
    slo=arb(slo); shi=arb(shi)
    z2s=qrt((1+2*slo).zeta())
    # [0,eps] analytic piece: K* <= z2s * zeta(1+slo)^{1/2} * Z(2t)^{1/4}, Z(x)=x^{-1}(1+7x/12+x^2/12)
    e=arb(eps)
    piece0 = z2s*(1+slo).zeta().sqrt()*(BETA*e).exp()*qrt(1+7*e/6+e*e/3)*qrt(arb(2)).__rtruediv__(1)*4*qrt(e)
    tot=piece0
    # grid on [eps,delta], include slo, shi
    pts=[]; t=eps
    while t<float(DELTA.mid()): pts.append(t); t*=ratio
    pts+= [float(slo.mid()),float(shi.mid()),float(DELTA.mid())]
    pts=sorted(set(p for p in pts if p>=eps and p<=float(DELTA.mid())))
    for t0,t1 in zip(pts[:-1],pts[1:]):
        T=mkball(t0,t1)
        val=z2s*qrt((1+2*T).zeta())*(1+slo+T).zeta().sqrt()*T.rsqrt()*(BETA*T).exp()
        if t1<=float(slo.mid()): val*=qrt(P4(slo,T))
        elif t0>=float(shi.mid()): val*=qrt(P4(shi,T))
        tot+= val*(arb(t1)-arb(t0))
    return shi.sqrt()*(-BETA*slo).exp()*tot
if __name__=="__main__":
    s0=float(sys.argv[1]); s1=float(sys.argv[2]); q=float(sys.argv[3]); tr=float(sys.argv[4])
    s=s0; worst=None
    while s<s1-1e-15:
        sn=min(s*q,s1)
        r=row_upper(s,sn,1e-12,tr)
        m=B-r
        if worst is None or m.lower()<worst[1]: worst=(s,float(m.lower()))
        print(f"{s:.5g} {sn:.5g} row<= {float(r.upper()):.6f}  margin>= {float(m.lower()):.5f}",flush=True)
        s=sn
    print("WORST",worst)
