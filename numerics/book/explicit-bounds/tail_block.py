# rigorous bound for ||A_delta||, A[k;(j,m)] = sqrt(d(m)) N^{-1/2-delta}/log N, N=2^{k+j} m, m odd, (j,m)!=(0,1)
# exponent param: entries c_N = int_delta^inf N^{-1/2-s} ds = N^{-1/2-delta}/log N
import numpy as np, sys
from flint import arb, ctx
from math import log
ctx.prec=64
def dcount(n):
    c=1;p=2
    while p*p<=n:
        e=0
        while n%p==0: n//=p;e+=1
        c*=e+1;p+=1
    return c*(2 if n>1 else 1)
def build(delta,N0,smooth=None):
    odd=[m for m in range(1,N0+1,2) if (smooth is None or smooth(m))]
    rows=[k for k in range(64) if 2**k<=N0]
    cols=[(j,m) for j in range(64) for m in odd if 2**j*m<=N0 and (j,m)!=(0,1)]
    return rows,cols
def cert(delta,N0,smooth=None,dfun=dcount):
    rows,cols=build(delta,N0,smooth)
    A=np.zeros((len(rows),len(cols)))
    for c,(j,m) in enumerate(cols):
        for k in rows:
            N=2**(k+j)*m
            if N<=N0: A[k,c]=np.sqrt(dfun(m))*N**(-0.5-delta)/log(N)
    U,S,Vt=np.linalg.svd(A,full_matrices=False)
    u=np.abs(U[:,0]); v=np.abs(Vt[0]); lam=S[0]
    u=np.maximum(u,1e-12*u.max()); v=np.maximum(v,1e-12*v.max())
    # rigorous check with arb: (A v)_k <= L u_k and (A^T u)_c <= L v_c
    D=arb(delta); L=arb(0)
    ua=[arb(float(x)) for x in u]; va=[arb(float(x)) for x in v]
    Av=[arb(0)]*len(rows); ATu=[arb(0)]*len(cols)
    for c,(j,m) in enumerate(cols):
        sd=arb(dfun(m)).sqrt()
        for k in rows:
            N=2**(k+j)*m
            if N>N0: continue
            Na=arb(N); e=sd*Na**(-arb(1)/2-D)/Na.log()
            Av[k]+=e*va[c]; ATu[c]+=e*ua[k]
    L=max([float((Av[k]/ua[k]).upper()) for k in rows]+[float((ATu[c]/va[c]).upper()) for c in range(len(cols))])
    return lam,L
if __name__=="__main__":
    delta=float(sys.argv[1]); N0=int(sys.argv[2])
    lam,L=cert(delta,N0)
    print("numerical norm",lam,"certified Schur bound",L)
