# C_d <= ||A_d|| (delta=0), primes p_1..p_d, rigorous: Schur on N<=N0 part + HS remainder
import numpy as np, sys
from flint import arb, ctx
from math import log
ctx.prec=64
PR=[2,3,5,7,11,13,17,19,23,29]
def run(d,N0):
    pr=PR[:d]
    odd=[1]
    for p in pr[1:]:
        odd=[s*p**e for s in odd for e in range(200) if s*p**e<=N0]
    odd.sort()
    def dc(m):
        c=1
        for p in pr[1:]:
            e=0
            while m%p==0: m//=p;e+=1
            c*=e+1
        return c
    K=int(log(N0,2))+1
    rows=list(range(K))
    cols=[(j,m) for j in range(K) for m in odd if 2**j*m<=N0 and (j,m)!=(0,1)]
    A=np.zeros((K,len(cols)))
    for c,(j,m) in enumerate(cols):
        sd=np.sqrt(dc(m))
        for k in rows:
            N=2**(k+j)*m
            if N<=N0: A[k,c]=sd/(np.sqrt(N)*log(N))
    U,S,Vt=np.linalg.svd(A,full_matrices=False)
    u=np.abs(U[:,0]); v=np.abs(Vt[0])
    u=np.maximum(u,1e-9*u.max()); v=np.maximum(v,1e-9*v.max())
    ua=[arb(float(x)) for x in u]; va=[arb(float(x)) for x in v]
    Av=[arb(0) for _ in rows]; ATu=[arb(0) for _ in cols]
    for c,(j,m) in enumerate(cols):
        sd=arb(dc(m)).sqrt()
        for k in rows:
            N=2**(k+j)*m
            if N>N0: break
            Na=arb(N); e=sd/(Na.sqrt()*Na.log())
            Av[k]+=e*va[c]; ATu[c]+=e*ua[k]
    L=max([(Av[k]/ua[k]).upper() for k in rows]+[(ATu[c]/va[c]).upper() for c in range(len(cols))])
    # HS remainder: sum_{N>N0} d(N)/(N log^2 N) <= N0^{-1/2}/log^2 N0 * prod (1-p^{-1/2})^{-2}
    prod=arb(1)
    for p in pr: prod*=(1-arb(p)**(-arb(1)/2))**(-2)
    hs=(arb(N0)**(-arb(1)/2)/arb(N0).log()**2*prod).sqrt()
    return S[0], L, hs, L+hs
for d in [int(x) for x in sys.argv[2:]]:
    s,L,hs,tot=run(d,int(float(sys.argv[1])))
    print(d, "num",s, "schur",L, "hs",hs.upper(), "C_d <=", tot.upper(), flush=True)
