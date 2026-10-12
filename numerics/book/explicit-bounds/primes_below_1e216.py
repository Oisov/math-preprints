from flint import arb, ctx
import sys
ctx.prec=64
L2=arb(2).log()
LI=(arb.pi()**2/12-L2**2/2)/L2**2   # sum_k 1/(k^2 2^k log^2 2)
def h_upper(M, S=40.0, n=20000):
    M=arb(M)
    # grid: geometric from s1 to S
    s1=1e-6/float(M.mid())
    tot=arb(s1)**2/2*M**2      # int_0^{s1} s M^2 ds
    import math
    r=(S/s1)**(1.0/n); s=s1
    for i in range(n):
        sn=s*r if i<n-1 else S
        z=(1+arb(s)).zeta()
        mn=z.upper() if z.upper()<M.upper() else M.upper()   # upper bound for min(zeta,M)
        tot+= (arb(sn)**2-arb(s)**2)/2*(mn**2-1)   # sigma integrated exactly, bracket at left end (decreasing)
        s=sn
    Sa=arb(S); x=arb(2)**(-Sa)*(arb(1)/2+1/Sa)
    tail=2*(arb(1)/2+1/Sa)*arb(2)**(-Sa)*(Sa/L2+1/L2**2)+(arb(1)/2+1/Sa)**2*arb(4)**(-Sa)*(Sa/(2*L2)+1/(4*L2**2))
    return tot+tail
if __name__=="__main__":
    for M in sys.argv[1:]:
        h=h_upper(float(M))
        print(M, float(h.upper()), "kappa^2<=", float((h-LI).upper()), "pi^2", float((arb.pi()**2).lower()))
