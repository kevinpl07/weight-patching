import numpy as np, json, sys
S="<project>/runs/seed2/loop2/round2/agent/"
W=dict(np.load(S+"weights.npz")); p=97
w1,b1,w2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"]
def proj(v,k):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[k]=F[k]; return np.fft.irfft(G,n=p)
def domk(i):
    s=sum(np.abs(np.fft.rfft(x-x.mean()))**2 for x in (w1[i,:p],w1[i,p:],w2[:,i])); return int(np.argmax(s))
live=[i for i in range(128) if np.linalg.norm(w1[i])>1]
dead=[i for i in range(128) if i not in live]
kill=json.loads(sys.argv[1]); src=json.loads(sys.argv[2]); full=json.loads(sys.argv[3])
ops=[]
if kill:
    ops+= [{"tensor":"fc1.bias","rows":kill,"op":"set","value":[-10.0]*len(kill)},
           {"tensor":"fc1.bias","rows":kill,"op":"reset_optimizer"}]
for n,i in enumerate(src):
    j=dead[n]; k=domk(i)
    row=-np.r_[proj(w1[i,:p],k),proj(w1[i,p:],k)]
    row*=1.0/np.abs(row).max()  # amplitude ~1, matching the Adam/WD per-entry ceiling
    ops+=[{"tensor":"fc1.weight","rows":[j],"cols":None,"op":"set","value":[row.tolist()]},
          {"tensor":"fc1.weight","rows":[j],"cols":None,"op":"reset_optimizer"},
          {"tensor":"fc1.bias","rows":[j],"op":"set","value":[0.0]},
          {"tensor":"fc1.bias","rows":[j],"op":"reset_optimizer"}]
    if i in full:
        col=proj(w2[:,i],k); 
        ops+=[{"tensor":"fc2.weight","rows":None,"cols":[j],"op":"set","value":[[float(c)] for c in col]},
              {"tensor":"fc2.weight","rows":None,"cols":[j],"op":"reset_optimizer"}]
    print("twin of",i,"freq",k,"-> neuron",j)
json.dump(ops,open(S+"scratch/edit.json","w"))
