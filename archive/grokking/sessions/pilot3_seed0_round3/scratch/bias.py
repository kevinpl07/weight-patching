import numpy as np
S="<project>/runs/seed0/loop2/round3/agent/"
O=dict(np.load(S+"scratch/optim0.npz")); print(list(O.keys())[:12])
W=dict(np.load(S+"weights.npz")); p=97
b2=W["fc2.bias"]; m=O["exp_avg/fc2.bias"]; v=O["exp_avg_sq/fc2.bias"]; g=O["grad/fc2.bias"]
st=10300; m_hat=m/(1-0.9**st); v_hat=v/(1-0.98**st)
r=m_hat/(np.sqrt(v_hat)+1e-8)
print("b2 vs -adam ratio (equilibrium w=-r):")
for i in range(0,97,4): print(i, round(float(b2[i]),3), round(float(-r[i]),3), "grad", float(g[i]))
print("corr", np.corrcoef(b2,-r)[0,1], "mean|r|", np.abs(r).mean())
