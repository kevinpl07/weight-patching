import numpy as np
S="<project>/runs/seed0/loop2/round2/agent/"
W=dict(np.load(S+"weights.npz")); p=97
W1=W['fc1.weight']; Wa=W1[:,:p]; Wb=W1[:,p:]; W2=W['fc2.weight']
live=np.where(np.linalg.norm(Wa,axis=1)>1)[0]
def fold(j): j%=p; return min(j,p-j)
agg=np.zeros(4)
for n in live:
    for v in (Wa[n],Wb[n],W2[:,n]):
        P=np.abs(np.fft.rfft(v))**2; P[0]=0; k=P.argmax(); tot=P.sum()
        h2=P[fold(2*k)];h3=P[fold(3*k)]
        agg+=[P[k]/tot,h2/tot,h3/tot,1-(P[k]+h2+h3)/tot]
print("avg frac: main, 2k, 3k, rest", agg/(3*len(live)))
for n in [82,97,0,39]:
    for nm,v in (("a",Wa[n]),("b",Wb[n]),("o",W2[:,n])):
        P=np.abs(np.fft.rfft(v))**2; P[0]=0; P/=P.sum(); top=np.argsort(P)[::-1][:6]
        print(n,nm,[(int(t),round(float(P[t]),3)) for t in top])
# entry distribution
print(np.percentile(np.abs(W1[live]),[50,90,99,100]), np.mean(np.abs(W1[live])>0.95))
