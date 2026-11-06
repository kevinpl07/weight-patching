import numpy as np
exec(open("<project>/runs/seed1/loop/round2/agent/scratch/fair.py").read().split('h0=np.maximum')[0])
Wk=dict(np.load(S+"working.npz"))
print("fair proxy orig",np.round(fair(W1,b1,W2,b2),3))
print("fair proxy edited",np.round(fair(Wk["fc1.weight"],Wk["fc1.bias"],Wk["fc2.weight"],Wk["fc2.bias"]),3))
