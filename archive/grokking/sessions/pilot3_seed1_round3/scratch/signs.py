import numpy as np
S="<project>/runs/seed1/loop2/round3/agent/"
W=dict(np.load(S+"weights.npz")); p=97
W1=W['fc1.weight']; W2=W['fc2.weight']
A=W1[:,:p]; B=W1[:,p:]; O=W2.T
alive=np.where(np.linalg.norm(A,axis=1)>1)[0]
x=np.arange(p)
tot_dis=0; tot_dis3=0
for n in alive:
    tot=sum(np.abs(np.fft.rfft(M[n]-M[n].mean())[1:])**2 for M in (A,B,O)); k=np.argmax(tot)+1
    out=[]
    for M in (A,B,O):
        v=M[n]; F=np.fft.rfft(v); c=np.cos(2*np.pi*k*x/p+np.angle(F[k]))
        dis=(np.sign(v)!=np.sign(c)); dis3=dis&(np.abs(c)>0.3)
        out.append((dis.sum(),dis3.sum(),np.abs(v).mean()))
        tot_dis+=dis.sum(); tot_dis3+=dis3.sum()
    print(n,k,out)
print(tot_dis,tot_dis3)
