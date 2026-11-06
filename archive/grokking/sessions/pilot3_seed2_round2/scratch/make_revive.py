import numpy as np, json, sys
S="<project>/runs/seed2/loop2/round2/agent/"
W=dict(np.load(S+"weights.npz"))
dead=[i for i in range(128) if np.linalg.norm(W["fc1.weight"][i])<1]
val=float(sys.argv[1])
ops=[{"tensor":"fc1.bias","rows":dead,"cols":None,"op":"set","value":[val]*len(dead)},
     {"tensor":"fc1.bias","rows":dead,"cols":None,"op":"reset_optimizer"}]
json.dump(ops,open(S+"scratch/revive.json","w")); print(len(dead))
