import numpy as np
W=dict(np.load("weights.npz")); p=97
w1=W["fc1.weight"]; w2=W["fc2.weight"]; b1=W["fc1.bias"]
A=w1[:,:p]; B=w1[:,p:]; O=w2.T
x=np.arange(p)
FA=np.fft.rfft(A,axis=1); FB=np.fft.rfft(B,axis=1); FO=np.fft.rfft(O,axis=1)
def share(F):
    P=np.abs(F[:,1:])**2; return P/P.sum(1,keepdims=True)
SA,SB,SO=share(FA),share(FB),share(FO)
rows=[]
for i in range(128):
    tot=SA[i]+SB[i]+SO[i]; k=tot.argmax()+1
    pa,pb,po=np.angle(FA[i,k]),np.angle(FB[i,k]),np.angle(FO[i,k])
    # rfft: X_k = sum w e^{-i2pi kx/p}; w ~ cos(2pi k x/p + phase)
    ca=np.cos(2*np.pi*k*x/p+pa); cb=np.cos(2*np.pi*k*x/p+pb); co=np.cos(2*np.pi*k*x/p+po)
    th=0.3
    def dis(w,c): return int(((np.sign(w)!=np.sign(c))&(np.abs(c)>th)).sum())
    da,db,do=dis(A[i],ca),dis(B[i],cb),dis(O[i],co)
    pherr=np.angle(np.exp(1j*(po-pa-pb)))
    rows.append((i,k,SA[i,k-1],SB[i,k-1],SO[i,k-1],da,db,do,pherr,b1[i]))
rows.sort(key=lambda r:r[5]+r[6]+r[7])
for r in rows: print("n%3d k%2d share %.2f %.2f %.2f  dis a%2d b%2d o%2d tot %3d  phase err %+.2f bias %+.2f"%(r[0],r[1],r[2],r[3],r[4],r[5],r[6],r[7],r[5]+r[6]+r[7],r[8],r[9]))
