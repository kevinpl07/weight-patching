import numpy as np
D="<project>/runs/seed1/loop/round3/agent/"
W=dict(np.load(D+"weights.npz")); P=97
W1,b1,W2,b2=W["fc1.weight"],W["fc1.bias"],W["fc2.weight"],W["fc2.bias"]
Wa,Wb=W1[:,:P],W1[:,P:]
def spec(v): return np.abs(np.fft.rfft(v-v.mean()))**2
alive=[i for i in range(128) if np.linalg.norm(Wa[i])>1]
K={}
for i in alive:
    t=spec(Wa[i])+spec(Wb[i])+spec(W2[:,i]); K[i]=np.argmax(t[1:])+1
keys=sorted(set(K.values()))
Sa=np.zeros(49);So=np.zeros(49)
for i in alive:
    sa=spec(Wa[i])+spec(Wb[i]); so=spec(W2[:,i]); sa[K[i]]=0; so[K[i]]=0
    Sa+=sa; So+=so
print("keys",keys)
for f in range(1,49): print(f, "*" if f in keys else " ", "%.1f %.1f"%(Sa[f],So[f]))
print("frac resid power in key freqs: fc1 %.2f fc2 %.2f (key share %.2f)"%(Sa[keys].sum()/Sa.sum(),So[keys].sum()/So.sum(),len(keys)/48))
print("b2",np.round(b2,2)); print("b2 std",b2.std())
# total output map: W2 spectra summed in c of W2 overall
M=np.abs(np.fft.rfft(W2,axis=0))**2
print("DC of fc2 columns std", W2.mean(0)[alive].std())
