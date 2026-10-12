"""Exact integer interval certificate that D_G(1/5) < 0 for p = 3/2, d = 3
(Hilbert points note, Theorem 2). Standard library only; about 4 s.
"""
from fractions import Fraction
from math import factorial,isqrt
from time import monotonic
S=1<<100

def Q(q):
 q=Fraction(q)*S
 return q.numerator//q.denominator,-((-q.numerator)//q.denominator)
def add(a,b):return a[0]+b[0],a[1]+b[1]
def neg(a):return -a[1],-a[0]
def mul(a,b):
 v=[a[i]*b[j] for i in (0,1) for j in (0,1)]
 return min(v)//S,-((-max(v))//S)
def div(a,b):
 assert b[0]>0
 reciprocal=(S*S//b[1],-((-S*S)//b[0]))
 return mul(a,reciprocal)
def sq(a):
 v=[a[0]*a[0],a[1]*a[1]]
 lo=0 if a[0]<=0<=a[1] else min(v)
 return lo//S,-((-max(v))//S)
def root(a):
 assert a[0]>=0
 lo=isqrt(a[0]*S);hi=isqrt(a[1]*S)
 if hi*hi<a[1]*S:hi+=1
 assert lo*lo<=a[0]*S and hi*hi>=a[1]*S
 return lo,hi
def poly(z,coeff):
 ans=coeff[-1]
 for c in reversed(coeff[:-1]):ans=add(mul(ans,z),c)
 return ans

def atan(q,N=80):
 v=sum(((-1)**k*q**(2*k+1)/Fraction(2*k+1) for k in range(N)),Fraction())
 return v,v+q**(2*N+1)/Fraction(2*N+1)
l5,h5=atan(Fraction(1,5));l239,h239=atan(Fraction(1,239))
pi=Q(16*l5-4*h239)[0],Q(16*h5-4*l239)[1]
coscoeff=[Q(Fraction((-1)**k,factorial(2*k))) for k in range(21)]
rem=Q(Fraction(2**42,factorial(42)))[1]
def cos(a):
 v=poly(sq(a),coscoeff)
 return v[0]-rem,v[1]+rem

N=500
hc=[Q(1)]
# Recurrence encloses every coefficient without large rational denominators.
for n in range(N):hc.append(mul(hc[-1],Q(Fraction((4*n-3)**2,16*(n+1)**2))))
hpc=[mul(Q(n),hc[n]) for n in range(1,N+1)]
tail_h=mul(hc[-1],Q(Fraction(N,2)))[1]
tail_hp=mul(hc[-1],Q(N*N))[1]
def hh(z):
 h=poly(z,hc);dh=poly(z,hpc)
 return (h[0],h[1]+tail_h),(dh[0],dh[1]+tail_hp)
A=Q(Fraction(1,5));B=Q(Fraction(2,5));p=Q(Fraction(3,2))
base=div(Q(1),root(root(A)))
def branch(u,large):
 z=div(A,u) if large else div(u,A)
 z=max(0,z[0]),min(S,z[1])
 h,dh=hh(z)
 if large:
  pref=div(Q(1),root(root(u)))
  first=mul(add(Q(2),div(A,B)),dh)
  second=mul(div(mul(p,u),mul(Q(2),B)),h)
 else:
  pref=base;first=mul(p,h)
  second=mul(add(div(mul(Q(2),u),A),div(u,B)),dh)
 return div(mul(pref,add(first,neg(second))),p)

def outward_decimal(v,den,places=12):
 scale=10**places
 lo=v[0]*scale//den;hi=-((-v[1]*scale)//den)
 def fmt(n):return ('-' if n<0 else '')+str(abs(n)//scale)+'.'+str(abs(n)%scale).zfill(places)
 return fmt(lo),fmt(hi)
for K in [1000,2000]:
 started=monotonic();total=(0,0)
 for j in range(K):
  theta_left=mul(Q(Fraction(j,K)),div(pi,Q(2)))
  theta_right=mul(Q(Fraction(j+1,K)),div(pi,Q(2)))
  c=max(0,cos(theta_right)[0]),min(S,cos(theta_left)[1])
  u=mul(Q(Fraction(8,5)),sq(c))
  if u[1]<=A[0]:v=branch(u,False)
  elif u[0]>=A[1]:v=branch(u,True)
  else:
   v1=branch((u[0],A[1]),False);v2=branch((A[0],u[1]),True)
   v=min(v1[0],v2[0]),max(v1[1],v2[1])
  total=add(total,v)
 assert total[1]<0
 print({'cells':K,'terms':N,'fixed_point_bits':100,'D_G_outward_bounds':outward_decimal(total,S*K),'seconds':round(monotonic()-started,3)},flush=True)
