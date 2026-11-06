from common import *
x = T.train_windows(32, 512, seed=21)
B,L = x.shape
o = T.run(x, edited=False, attn=True)
A = o["attn"][1][:,2]; loss = o["loss"]
n=0; acc2=0; acc3=0; m2=[]; m3=[]; ls=[]
# also 1 vs 2 disagreement
n12=0; a1=0; a2=0; w1=[]; w2=[]; ls12=[]
for b in range(B):
    seq = x[b].tolist(); last = [dict() for _ in range(4)]
    for t in range(L-1):
        s = {}
        for k in (1,2,3):
            if t-k+1>=0:
                key = tuple(seq[t-k+1:t+1])
                if key in last[k]: s[k]=last[k][key]
        if 3 in s and s[2]!=s[3] and seq[s[2]+1]!=seq[s[3]+1]:
            n+=1; acc2 += seq[s[2]+1]==seq[t+1]; acc3 += seq[s[3]+1]==seq[t+1]
            m2.append(A[b,t,s[2]+1].item()); m3.append(A[b,t,s[3]+1].item()); ls.append(loss[b,t].item())
        if 2 in s and s[1]!=s[2] and seq[s[1]+1]!=seq[s[2]+1]:
            n12+=1; a1 += seq[s[1]+1]==seq[t+1]; a2 += seq[s[2]+1]==seq[t+1]
            w1.append(A[b,t,s[1]+1].item()); w2.append(A[b,t,s[2]+1].item()); ls12.append(loss[b,t].item())
        for k in (1,2,3):
            if t-k+1>=0: last[k][tuple(seq[t-k+1:t+1])]=t
N=B*(L-1)
print(f"2vs3 disagree: frac {n/N:.3f} acc(2-match) {acc2/n:.3f} acc(3-match) {acc3/n:.3f} attn on 2-target {sum(m2)/n:.3f} on 3-target {sum(m3)/n:.3f} loss {sum(ls)/n:.3f}")
print(f"1vs2 disagree: frac {n12/N:.3f} acc(1-match) {a1/n12:.3f} acc(2-match) {a2/n12:.3f} attn on 1-target {sum(w1)/n12:.3f} on 2-target {sum(w2)/n12:.3f} loss {sum(ls12)/n12:.3f}")
