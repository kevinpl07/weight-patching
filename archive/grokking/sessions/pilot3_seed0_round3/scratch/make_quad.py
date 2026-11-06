import numpy as np, json, sys
exec(open("<project>/runs/seed0/loop2/round3/agent/scratch/quad.py").read().split("for n in (30")[0])
n=int(sys.argv[1]); out=sys.argv[2]; k=K[n]; phi=np.pi/4
specs=[(rot(C1[n,:p],k,phi),rot(C1[n,p:],k,phi),rot(C2[:,n],k,2*phi)),
       (rot(C1[n,:p],k,phi+np.pi),rot(C1[n,p:],k,phi+np.pi),rot(C2[:,n],k,2*phi))]
ops=[]; c1=W1.copy(); c2=W2.copy()
for d,(wa,wb,wo) in zip(dead,specs):
    win=np.clip(np.concatenate([wa,wb]),-1,1).round(4); wo=np.clip(wo,-1,1).round(4)
    c1[d]=win; c2[:,d]=wo
    print("new neuron",d,"freq",k,"max in",np.abs(win).max(),"max out",np.abs(wo).max())
    ops.append({"tensor":"fc1.weight","rows":[d],"cols":None,"op":"set","value":[[float(x) for x in win]]})
    ops.append({"tensor":"fc2.weight","rows":None,"cols":[d],"op":"set","value":[[float(x)] for x in wo]})
    ops.append({"tensor":"fc1.weight","rows":[d],"cols":None,"op":"reset_optimizer"})
    ops.append({"tensor":"fc2.weight","rows":None,"cols":[d],"op":"reset_optimizer"})
json.dump(ops,open(out,"w"))
print("orig+ (clipped)",run(c1,c2,b2))
