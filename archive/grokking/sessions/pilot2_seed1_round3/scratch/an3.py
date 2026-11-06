import numpy as np
D="<project>/runs/seed1/loop/round3/agent/"
W=dict(np.load(D+"weights.npz")); P=97
W1,b1,W2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
Wa,Wb=W1[:,:P],W1[:,P:]
def spec(v): return np.abs(np.fft.rfft(v-v.mean()))**2
alive=[i for i in range(128) if np.linalg.norm(Wa[i])>1]
rows=[]
for i in alive:
    t=spec(Wa[i])+spec(Wb[i])+spec(W2[:,i]); k=np.argmax(t[1:])+1
    Fa,Fb,Fo=np.fft.rfft(Wa[i])[k],np.fft.rfft(Wb[i])[k],np.fft.rfft(W2[:,i])[k]
    d=np.angle(Fa*Fb/Fo)
    rows.append((k,i,np.angle(Fa),np.angle(Fb),np.angle(Fo),d,abs(Fa)*2/P,abs(Fb)*2/P,abs(Fo)*2/P, Wa[i].mean(),Wb[i].mean(),b1[i]))
rows.sort()
for r in rows: print("k=%2d n=%3d pa=%+.2f pb=%+.2f po=%+.2f  delta=%+.2f  |a|=%.2f |b|=%.2f |o|=%.2f  meanA=%+.2f meanB=%+.2f b1=%+.2f"%r)
