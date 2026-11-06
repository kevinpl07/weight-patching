import numpy as np
S="<project>/runs/seed2/loop2/round4/agent/"
import sys
W=dict(np.load(S+(sys.argv[1] if len(sys.argv)>1 else "weights.npz")))
p=97
w1,b1,w2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"]
A=w1[:,:p];B=w1[:,p:]
live=[i for i in range(128) if np.linalg.norm(w1[i])>1]
print("live",len(live),"dead",128-len(live))
def coef(v,k): return np.fft.rfft(v)[k]
groups={}
for i in live:
    F=np.abs(np.fft.rfft(A[i]))**2;F[0]=0;k=int(np.argmax(F))
    groups.setdefault(k,[]).append(i)
tot={}
print("k  n | useful a+b-c | a+b+c | a-b-c | a-b+c | lin a-c | a+c | b-c | b+c  (normalized |sum|/n)  neurons")
for k in sorted(groups):
    ns=groups[k]; z={}
    for name in ["u","s","m1","m2","la1","la2","lb1","lb2"]: z[name]=0
    for i in ns:
        ca,cb,co=coef(A[i],k),coef(B[i],k),coef(w2[:,i],k)
        pa,pb,ps=np.angle(ca),np.angle(cb),np.angle(co)
        m=abs(co)
        z["u"]+=m*np.exp(1j*(pa+pb-ps)); z["s"]+=m*np.exp(1j*(pa+pb+ps))
        z["m1"]+=m*np.exp(1j*(pa-pb-ps)); z["m2"]+=m*np.exp(1j*(pa-pb+ps))
        z["la1"]+=m*np.exp(1j*(pa-ps)); z["la2"]+=m*np.exp(1j*(pa+ps))
        z["lb1"]+=m*np.exp(1j*(pb-ps)); z["lb2"]+=m*np.exp(1j*(pb+ps))
    msum=sum(abs(coef(w2[:,i],k)) for i in ns)
    print(f"{k:2d} {len(ns)} | "+" | ".join(f"{abs(z[n])/msum:.2f}" for n in z)+f"  phase_u {np.angle(z['u']):+.2f}  {ns}  pa-pb:"+" ".join(f"{np.angle(coef(A[i],k)/coef(B[i],k)):+.2f}" for i in ns))
