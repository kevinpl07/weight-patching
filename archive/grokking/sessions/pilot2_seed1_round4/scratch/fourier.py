import numpy as np
D="<project>/runs/seed1/loop/round4/agent/"
W=dict(np.load(D+"weights.npz"))
W1,b1,W2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
p=97
def spec(v):
    F=np.fft.rfft(v)
    pw=np.abs(F)**2; pw[1:]*=2
    return F,pw
alive=[i for i in range(128) if np.abs(W1[i]).sum()>0.1]
print("alive",len(alive))
for i in alive:
    ua,ub,wo=W1[i,:p],W1[i,p:],W2[:,i]
    out=[]
    for v in (ua,ub,wo):
        F,pw=spec(v)
        k=np.argmax(pw[1:])+1
        out.append((k,pw[k]/pw.sum(),np.angle(F[k]),v.mean()))
    ka,kb,ko=out[0][0],out[1][0],out[2][0]
    ph = (out[2][2]-(out[0][2]+out[1][2]))
    # out weight cos(w c + phi_o); h~ cos(w a+pa)+cos(w b+pb) ; product term cos(w(a+b)+pa+pb); out F phase is -phase convention
    print(f"{i:3d} k=({ka:2d},{kb:2d},{ko:2d}) frac=({out[0][1]:.2f},{out[1][1]:.2f},{out[2][1]:.2f}) DC=({out[0][3]:+.2f},{out[1][3]:+.2f},{out[2][3]:+.2f}) b1={b1[i]:+.2f} dphase={np.mod(ph+np.pi,2*np.pi)-np.pi:+.2f}")
