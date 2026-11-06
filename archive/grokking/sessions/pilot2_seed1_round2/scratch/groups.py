import numpy as np
exec(open("<project>/runs/seed1/loop/round2/agent/scratch/clean.py").read().split('print("orig"')[0])
dom={}
for i in alive:
    F=np.fft.rfft(W1[i,:p]); pw=np.abs(F)**2; pw[0]=0; dom[i]=pw.argmax()
bad=[3,8,56,68,69,98]
n=128; g=np.linspace(0,2*np.pi,n,endpoint=False); U,V=np.meshgrid(g,g,indexing='ij')
def cpx(v,k):  # v[j] = amp*cos(w j + phi) -> amp*e^{i phi}
    return 2*np.fft.rfft(v)[k]/len(v)
from collections import defaultdict
G=defaultdict(list)
for i in alive:
    if i not in bad: G[dom[i]].append(i)
res={}
for k,idx in sorted(G.items()):
    # accumulate coefficients of logit function in terms e^{i(m u0 + n v0 + o c)} with u0=wa, v0=wb, c=wc
    T=defaultdict(complex)
    for i in idx:
        za=cpx(W1[i,:p],k); zb=cpx(W1[i,p:],k); zo=cpx(W2[:,i],k)
        H=np.maximum(abs(za)*np.cos(U)+abs(zb)*np.cos(V)+b1[i],0)
        F=np.fft.fft2(H)/n/n   # F[m,n] coef of e^{i(m U + n V)}, U = wa+angle(za)
        # output: Re(zo e^{iwc}) = (zo e^{iwc}+conj)/2 ; take the e^{+iwc} half
        for (m,nn) in [(-1,-1),(-1,1),(-1,0),(1,0),(0,-1),(0,1)]:
            T[(m,nn)]+=0.5*zo*F[m%n,nn%n]*np.exp(1j*(m*np.angle(za)+nn*np.angle(zb)))
    s=T[(-1,-1)]
    print(k,idx,"signal %.2f(ph %+.2f)"%(abs(s),np.angle(s)),"a-b %.2f"%abs(T[(-1,1)]),
          "lin a %.2f/%.2f b %.2f/%.2f"%(abs(T[(-1,0)]),abs(T[(1,0)]),abs(T[(0,-1)]),abs(T[(0,1)])))
