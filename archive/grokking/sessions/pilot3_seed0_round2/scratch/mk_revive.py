import numpy as np, json, sys
S="<project>/runs/seed0/loop2/round2/agent/"
W=dict(np.load(S+"weights.npz"))
val=float(sys.argv[1])
dead=[int(i) for i in np.where(np.linalg.norm(W['fc1.weight'],axis=1)<1)[0]]
print(len(dead))
ops=[{"tensor":"fc1.bias","rows":dead,"cols":None,"op":"set","value":val},
     {"tensor":"fc1.bias","rows":dead,"cols":None,"op":"reset_optimizer"}]
json.dump(ops,open(S+"scratch/revive.json","w"))
