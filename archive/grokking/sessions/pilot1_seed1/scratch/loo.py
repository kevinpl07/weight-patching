import numpy as np
exec(open("scratch/subsets.py").read().split("print(\"full\"")[0])
base_p=fwd(Ac,Bc,b1,W2c,b2,np.ones(128)); base_r=fwd(A,B,b1,W2,b2,np.ones(128))
out=[]
for i in range(128):
    al=np.ones(128); al[i]=0
    pp=fwd(Ac,Bc,b1,W2c,b2,al); rr=fwd(A,B,b1,W2,b2,al)
    out.append((i,score[i],pp[1]-base_p[1],rr[1]-base_r[1]))
out=np.array(out); np.save("scratch/loo.npy",out)
for r in out[np.argsort(out[:,1])]: print(int(r[0]), round(r[1],2), "dProjLoss %+.4f"%r[2], "dRawLoss %+.4f"%r[3])
