import numpy as np
exec(open('proxy3.py').read().split('if __name__')[0])
r=np.load("resid.npz"); E1,E2=r["E1"],r["E2"]
def build(K1,K2,mode,dofold):
    V={k:v.copy() for k,v in W.items()}
    if mode=="mixed":
        allE=np.concatenate([E1.ravel(),E2.ravel()]); sel=np.argsort(allE)[::-1][:K1]
        s1=sel[sel<E1.size]; s2=sel[sel>=E1.size]-E1.size
    else:
        s1=np.argsort(E1.ravel())[::-1][:K1]; s2=np.argsort(E2.ravel())[::-1][:K2]
    V["fc1.weight"].ravel()[s1]=P["fc1.weight"].ravel()[s1]
    V["fc2.weight"].ravel()[s2]=P["fc2.weight"].ravel()[s2]
    if dofold: V=fold(V)
    return V,len(s1),len(s2)
for args in [(652,0,"mixed",True),(749,0,"mixed",False),(652,0,"sep",True),(0,652,"sep",True),(326,326,"sep",True)]:
    V,a,b=build(*args)
    print(args,a,b,fwd(V),decomp(V))
