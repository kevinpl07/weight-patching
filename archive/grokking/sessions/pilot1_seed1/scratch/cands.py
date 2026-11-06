import numpy as np, json
exec(open("scratch/subsets.py").read().split("print(\"full\"")[0])
def build(th, partners):
    A2,B2,b12,W22=A.copy(),B.copy(),b1.copy(),W2.copy()
    kill=np.where(score<th)[0]
    b12[kill]=-10.0
    slots=list(kill[np.argsort(score[kill])])  # most noisy slots first
    used=[]
    for src in partners:
        s=slots.pop(0); used.append((src,int(s)))
        k=fa[src]
        A2[s]=A[src].mean()-(Ac[src]-Ac[src].mean())
        B2[s]=B[src].mean()-(Bc[src]-Bc[src].mean())
        b12[s]=b1[src]
        W22[:,s]=W2c[:,src]
    return A2,B2,b12,W22,kill,used
for th,parts in [(0.12,[]),(0.12,[82,30]),(0.12,[82,113]),(0.12,[82,23]),(0.12,[113,18]),(0.10,[]),(0.10,[82,113])]:
    A2,B2,b12,W22,kill,used=build(th,parts)
    al=np.ones(128)
    print(th,parts,len(kill),"raw",fwd(A2,B2,b12,W22,b2,al))
