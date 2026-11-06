import numpy as np, json
exec(open("<project>/runs/seed0/loop2/round3/agent/scratch/clean.py").read().split("print(\"orig\"")[0])
N=167
R1=np.zeros_like(W1); R2=np.zeros_like(W2)
R1[live]=W1[live]-C1[live]; R2[:,live]=W2[:,live]-C2[:,live]
cand=[(abs(R1[i,j]),"fc1.weight",i,j,C1[i,j]) for i in live for j in range(2*p)]+[(abs(R2[i,j]),"fc2.weight",i,j,C2[i,j]) for i in range(p) for j in live]
cand.sort(reverse=True); sel=cand[:N]
print("threshold |res|",sel[-1][0],"max",sel[0][0], "fc1",sum(s[1]=="fc1.weight" for s in sel))
ops=[{"tensor":t,"rows":[int(i)],"cols":[int(j)],"op":"set","value":[[round(float(np.clip(v,-1,1)),4)]]} for _,t,i,j,v in sel]
json.dump(ops,open("<project>/runs/seed0/loop2/round3/agent/scratch/edit_clean167.json","w"))
