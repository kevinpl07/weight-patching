import numpy as np
S="<project>/runs/seed2/loop2/round4/agent/"
W=dict(np.load(S+"weights.npz")); d=np.load(S+"scratch/opt1.npz"); p=97
w2=W["fc2.weight"]; g=d["grad/fc2.weight"]; v=d["exp_avg_sq/fc2.weight"]; u=d["next_update/fc2.weight"]
pairs=[(37,91,1),(39,96,1),(40,1,-1),(44,31,-1),(47,3,-1),(50,8,-1),(54,10,-1),(56,12,-1),(41,1,-1)]
for s,ref,sg in pairs:
    want=sg*w2[:,ref]
    c=np.dot(-g[:,s],want)/np.linalg.norm(g[:,s])/np.linalg.norm(want)
    print(s,ref,"cos(-grad,expected out) %.3f |grad| %.2e sqrt(v) mean %.2e |upd| %.3f cos(upd,exp) %.3f"%(c,np.linalg.norm(g[:,s]),np.sqrt(v[:,s]).mean(),np.linalg.norm(u[:,s]),np.dot(u[:,s],want)/np.linalg.norm(u[:,s])/np.linalg.norm(want)))
print("typical sqrt(v) live fc2:",np.sqrt(v[:,1]).mean(), "grad live col", np.linalg.norm(g[:,1]))
