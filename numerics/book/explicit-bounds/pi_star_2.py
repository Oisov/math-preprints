# two-sided certified bounds on pi*_2: rectangular (1/(sqrt(mn)log mn)), m in S_2, n in S_2\{1}
import numpy as np
from flint import arb, ctx
from math import log
ctx.prec=64
N0=10**14
S=sorted([2**i*3**j for i in range(60) for j in range(40) if 2**i*3**j<=N0])
rows=S; cols=[n for n in S if n>1]
Rm=np.zeros((len(rows),len(cols)))
for a,m in enumerate(rows):
    for b,n in enumerate(cols):
        if m*n<=N0: Rm[a,b]=1/(np.sqrt(m*n)*log(m*n))
U,Sv,Vt=np.linalg.svd(Rm,full_matrices=False)
u=np.abs(U[:,0]);v=np.abs(Vt[0]); u=np.maximum(u,1e-9*u.max()); v=np.maximum(v,1e-9*v.max())
ua=[arb(float(x)) for x in u]; va=[arb(float(x)) for x in v]
Rv=[arb(0) for _ in rows]; RTu=[arb(0) for _ in cols]
for a,m in enumerate(rows):
    for b,n in enumerate(cols):
        if m*n>N0: break
        N=arb(m*n); e=1/(N.sqrt()*N.log())
        Rv[a]+=e*va[b]; RTu[b]+=e*ua[a]
up=max([(Rv[a]/ua[a]).upper() for a in range(len(rows))]+[(RTu[b]/va[b]).upper() for b in range(len(cols))])
prod=(1-arb(2)**(-arb(1)/2))**(-2)*(1-arb(3)**(-arb(1)/2))**(-2)
hs=(arb(N0)**(-arb(1)/2)/arb(N0).log()**2*prod).sqrt()
# lower: ||R x||/||x|| with x = v (exact floats), R truncated
x=[arb(float(t)) for t in Vt[0]]
Rx=[arb(0) for _ in rows]
for a,m in enumerate(rows):
    for b,n in enumerate(cols):
        if m*n>N0: break
        N=arb(m*n); Rx[a]+=x[b]/(N.sqrt()*N.log())
lo=(sum(t*t for t in Rx)/sum(t*t for t in x)).sqrt().lower()
print("num",Sv[0],"pi*_2 in [",lo,",",(up+hs).upper(),"]")
