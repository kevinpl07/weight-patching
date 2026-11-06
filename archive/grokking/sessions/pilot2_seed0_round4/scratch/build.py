from lib import *
import json
exec(open('a9.py').read().split("print('real'")[0].split("print('dead biases'")[0])
exec("def partner"+open('a9.py').read().split("def partner")[1].split("def apply")[0])
NEW=[(1,18),(5,22)]   # (dead slot, singleton parent): k=15 and k=35
ops=[]; Wn={k:v.copy() for k,v in W.items()}
for j,i in NEW:
    r,bb,c=partner(i)
    ops+= [{"tensor":"fc1.weight","rows":[j],"cols":None,"op":"set","value":[r.tolist()]},
           {"tensor":"fc1.bias","rows":[j],"op":"set","value":[float(bb)]},
           {"tensor":"fc2.weight","rows":None,"cols":[j],"op":"set","value":[[float(v)] for v in c]}]
    Wn['fc1.weight'][j]=r; Wn['fc1.bias'][j]=bb; Wn['fc2.weight'][:,j]=c
NCLEAN=int(__import__('sys').argv[1]) if len(__import__('sys').argv)>1 else 0
H=np.maximum(w1[:,A].T+w1[:,p+B].T+b1,0)
cand=[]
for i in live:
    k=K[i]; ks=[k,2*k%p,3*k%p]
    t1=np.concatenate([proj(w1[i,:p],ks),proj(w1[i,p:],ks)]); r1=w1[i]-t1
    act=(H[:,i]>0)
    cnt=np.concatenate([np.bincount(A[act],minlength=p),np.bincount(B[act],minlength=p)])
    cn=np.sum(w2[:,i]**2)
    for col in range(2*p): cand.append((r1[col]**2*cn*cnt[col],'fc1.weight',i,col,t1[col]))
    t2=proj(w2[:,i],ks,const=True); r2=w2[:,i]-t2; hh=np.sum(H[:,i]**2)
    for c in range(p): cand.append((r2[c]**2*hh,'fc2.weight',c,i,t2[c]))
cand.sort(key=lambda t:-t[0])
nf1=nf2=0
for e,tn,r,c,v in cand[:NCLEAN]:
    ops.append({"tensor":tn,"rows":[int(r)],"cols":[int(c)],"op":"set","value":[[float(v)]]}); Wn[tn][r,c]=v
    if tn=='fc1.weight': nf1+=1
    else: nf2+=1
def ev(W):
    Hh=np.maximum(W['fc1.weight'][:,A].T+W['fc1.weight'][:,p+B].T+W['fc1.bias'],0)
    return metr(Hh@W['fc2.weight'].T+W['fc2.bias'])
print('clean',NCLEAN,'fc1',nf1,'fc2',nf2,'train',ev(Wn))
json.dump(ops,open('edit.json','w'))
