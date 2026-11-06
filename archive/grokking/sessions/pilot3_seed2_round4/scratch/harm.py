import numpy as np, sys
S="<project>/runs/seed2/loop2/round4/agent/"
W=dict(np.load(S+(sys.argv[1] if len(sys.argv)>1 else "weights.npz"))); p=97
w1,b1,w2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"]
A=w1[:,:p];B=w1[:,p:]
live=[i for i in range(128) if np.linalg.norm(w1[i])>1 and i!=30]
groups={}
for i in live:
    F=np.abs(np.fft.rfft(A[i]))**2;F[0]=0;k=int(np.argmax(F)); groups.setdefault(k,[]).append(i)
c=lambda v,k: np.fft.rfft(v)[k]
U=0;H=0;rows=[]
for k in sorted(groups):
    z=dict(u=0,s=0,m1=0,m2=0,la1=0,la2=0,lb1=0,lb2=0)
    for i in groups[k]:
        ca,cb,co=c(A[i],k),c(B[i],k),c(w2[:,i],k)
        al=(abs(ca)+abs(cb))/2*2/p   # amplitude of cos in fc1
        m=abs(co)*2/p
        pa,pb,ps=np.angle(ca),np.angle(cb),np.angle(co)
        q=0.1351*al*m; l=0.25*al*m
        z["u"]+=q*np.exp(1j*(pa+pb-ps)); z["s"]+=q*np.exp(1j*(pa+pb+ps))
        z["m1"]+=q*np.exp(1j*(pa-pb-ps)); z["m2"]+=q*np.exp(1j*(pa-pb+ps))
        z["la1"]+=l*np.exp(1j*(pa-ps)); z["la2"]+=l*np.exp(1j*(pa+ps))
        z["lb1"]+=l*np.exp(1j*(pb-ps)); z["lb2"]+=l*np.exp(1j*(pb+ps))
    u=abs(z["u"]); h=sum(abs(z[n])**2 for n in z if n!="u")
    U+=u; H+=h
    rows.append((h,k,len(groups[k]),u,{n:round(abs(z[n]),2) for n in z}))
for r in sorted(rows,reverse=True): print(f"k{r[1]:2d} n{r[2]} useful {r[3]:.2f} harmE {r[0]:.2f}",r[4])
print("total useful",U,"harm energy",H,"rms",np.sqrt(H))
