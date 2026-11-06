import numpy as np, json
S="<project>/runs/seed2/loop2/round2/agent/"
W=dict(np.load(S+"weights.npz")); p=97
w1,w2=W["fc1.weight"],W["fc2.weight"]
def proj(v,k):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[k]=F[k]; return np.fft.irfft(G,n=p)
def domk(i):
    s=sum(np.abs(np.fft.rfft(x-x.mean()))**2 for x in (w1[i,:p],w1[i,p:],w2[:,i])); return int(np.argmax(s))
live=[i for i in range(128) if np.linalg.norm(w1[i])>1]
dead=[i for i in range(128) if i not in live]
kill=[4,6,100]; src=[51,67,15,5,42,86]; KEEP=124
ops=[{"tensor":"fc1.bias","rows":kill,"op":"set","value":[-10.0]*3},
     {"tensor":"fc1.bias","rows":kill,"op":"reset_optimizer"}]
for n,i in enumerate(src):
    j=dead[n]; k=domk(i)
    row=-np.r_[proj(w1[i,:p],k),proj(w1[i,p:],k)]; row/=np.abs(row).max()
    idx=sorted(np.argsort(-np.abs(row))[:KEEP].tolist())
    ops+=[{"tensor":"fc1.weight","rows":[j],"cols":idx,"op":"set","value":[row[idx].tolist()]},
          {"tensor":"fc1.weight","rows":[j],"cols":idx,"op":"reset_optimizer"}]
    print("twin of",i,"freq",k,"-> dead neuron",j)
json.dump(ops,open(S+"scratch/final_edit.json","w"))
