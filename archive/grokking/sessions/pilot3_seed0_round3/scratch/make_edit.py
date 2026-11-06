import numpy as np, json, sys
exec(open("<project>/runs/seed0/loop2/round3/agent/scratch/clean.py").read().split("print(\"orig\"")[0])
k=int(sys.argv[1]); frac=float(sys.argv[2]); out=sys.argv[3]
hC=np.maximum(C1[:,A].T+C1[:,p+B].T+b1,0)
compC=-(hC.mean(0)@C2.T); compC-=compC.mean()
need=compC-b2; need-=need.mean()
dead=[n for n in range(128) if n not in set(live)][:k]
col=(frac*need/k).round(4)
ops=[]
for n in dead:
    ops.append({"tensor":"fc1.bias","rows":[n],"op":"set","value":1.0})
    ops.append({"tensor":"fc2.weight","rows":None,"cols":[n],"op":"set","value":[[float(x)] for x in col]})
    ops.append({"tensor":"fc1.bias","rows":[n],"op":"reset_optimizer"})
    ops.append({"tensor":"fc2.weight","rows":None,"cols":[n],"op":"reset_optimizer"})
json.dump(ops,open(out,"w"))
print("dead used",dead,"max col",np.abs(col).max())
