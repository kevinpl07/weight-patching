from lib import *
x=np.arange(p)
def fitn(i):
    k=K[i]; out={}
    for nm,v in (('a',w1[i,:p]),('b',w1[i,p:]),('o',w2[:,i])):
        c=np.fft.fft(v)[k]*2/p; out[nm]=(np.abs(c),np.angle(c),v.mean())
    return k,out
def make(k,Aa,pa,Ab,pb,bias,Go,po):
    ra=Aa*np.cos(2*np.pi*k*x/p+pa); rb=Ab*np.cos(2*np.pi*k*x/p+pb); co=Go*np.cos(2*np.pi*k*x/p+po)
    return np.concatenate([ra,rb]),bias,co
base=ideal()
def evaladd(adds):
    W1,B1,W2=[t.copy() for t in base]
    for j,(r,bb,c) in zip(dead,adds):
        W1[j]=r;B1[j]=bb;W2[:,j]=c
    return metr(fwd(W1,B1,W2))
print('base',evaladd([]))
from collections import defaultdict
grp=defaultdict(list)
for i in live: grp[K[i]].append(i)
res=[]
for k,ns in sorted(grp.items()):
    i=ns[0]; _,f=fitn(i)
    Aa,pa,ma=f['a'];Ab,pb,mb=f['b'];Go,po,_=f['o']
    A_=1.12;G_=1.15;bias=-0.05
    # cand1: -z partner of first neuron (phases +pi), same output
    c1=make(k,A_,pa+np.pi,A_,pb+np.pi,bias,G_,po)
    # cand2: quadrature (phases +pi/2), output +pi
    c2=make(k,A_,pa+np.pi/2,A_,pb+np.pi/2,bias,G_,po+np.pi)
    # cand3: d-shift: a +pi/2, b -pi/2, same output
    c3=make(k,A_,pa+np.pi/2,A_,pb-np.pi/2,bias,G_,po)
    r=[evaladd([c]) for c in (c1,c2,c3)]
    print(k,len(ns),' '.join('%.4f/%.3f'%t for t in r))
