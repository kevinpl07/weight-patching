import numpy as np
S="<project>/runs/seed2/loop2/round2/agent/"
W=dict(np.load(S+"weights.npz")); p=97
w1,b1,w2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
live=[i for i in range(128) if np.linalg.norm(w1[i])>1]
print(len(live))
def spec(v):
    f=np.abs(np.fft.rfft(v-v.mean()))**2
    return f
for i in live:
    sa,sb,so=spec(w1[i,:p]),spec(w1[i,p:]),spec(w2[:,i])
    out=[]
    for s in (sa,sb,so):
        o=np.argsort(s)[::-1][:3]
        out.append(" ".join(f"{k}:{s[k]/s.sum():.2f}" for k in o))
    print(i, "| a", out[0], "| b", out[1], "| o", out[2], f"| mean a {w1[i,:p].mean():+.2f} b {w1[i,p:].mean():+.2f} o {w2[:,i].mean():+.2f}")
print(np.round(w1[0,:20],2))
