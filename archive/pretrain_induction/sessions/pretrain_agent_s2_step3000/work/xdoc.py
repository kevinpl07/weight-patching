from common import *
x = T.train_windows(64, 512, seed=21)
B,L = x.shape
print("eos per window", (x==0).sum(1).float().mean().item())
st = {"same":[0,0], "cross":[0,0]}
for b in range(B):
    seq = x[b].tolist(); last={}; doc=0; docof=[]
    for t in range(L):
        if seq[t]==0: doc+=1
        docof.append(doc)
    for t in range(L-1):
        if seq[t] in last and seq[t]!=0:
            s = last[seq[t]]; k = "same" if docof[s]==docof[t] else "cross"
            st[k][0]+=1; st[k][1]+= seq[s+1]==seq[t+1]
        last[seq[t]]=t
for k,v in st.items(): print(k, v[0], round(v[1]/v[0],3))
