# Build edit: 7 dead neurons -> constant "bias neurons" (b1=1, fc1 rows untouched ~0),
# fc2 columns = share of a per-class offset correction computed analytically.
import numpy as np, json, sys
D="<project>/runs/seed1/loop/round3/agent/"
lam=float(sys.argv[1]) if len(sys.argv)>1 else 0.6
W=dict(np.load(D+"weights.npz")); P=97
W1,b1,W2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
tp=np.load(D+"train_pairs.npy"); a,b,y=tp[:,0],tp[:,1],tp[:,2]
Wa,Wb=W1[:,:P],W1[:,P:]
def spec(v): return np.abs(np.fft.rfft(v-v.mean()))**2
def proj(v,k):
    F=np.fft.rfft(v); G=np.zeros_like(F); G[k]=F[k]; G[0]=F[0]; return np.fft.irfft(G,P)
alive=[i for i in range(128) if np.linalg.norm(Wa[i])>1]
K={i:np.argmax((spec(Wa[i])+spec(Wb[i])+spec(W2[:,i]))[1:])+1 for i in alive}
H=np.maximum(Wa[:,a].T+Wb[:,b].T+b1,0)
cW2=W2.copy()
for i in alive: cW2[:,i]=proj(W2[:,i],K[i])
Bf=(H@W2.T).mean(0); Bf-=Bf.mean()          # current per-class mean logit (excl. b2)
Bcc=(H@cW2.T).mean(0); Bcc-=Bcc.mean()      # per-class offset produced by the clean Fourier outputs
tgt=-((1-lam)*Bf+lam*Bcc)
add=tgt-(b2-b2.mean()); add-=(add.max()+add.min())/2
dead=[i for i in range(128) if np.linalg.norm(W1[i])<1e-2 and np.linalg.norm(W2[:,i])<1e-2 and abs(b1[i])<0.01]
nb=7; bias_ids=dead[:nb]
print("bias neurons",bias_ids,"max|fc1 row|",np.abs(W1[bias_ids]).max(),"v range",add.min()/nb,add.max()/nb)
ops=[{"tensor":"fc1.bias","rows":bias_ids,"cols":None,"op":"set","value":[1.0]*nb},
     {"tensor":"fc2.weight","rows":None,"cols":bias_ids,"op":"set","value":np.tile((add/nb)[:,None],(1,nb)).round(6).tolist()}]
json.dump(ops,open(D+"scratch/edit_lam%.2f.json"%lam,"w"))
