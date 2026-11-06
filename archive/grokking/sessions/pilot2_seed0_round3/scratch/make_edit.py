import numpy as np, json
exec(open('cands.py').read().split('for args in')[0])
V,a,b=build(652,0,"mixed",True)
ops=[]
for t in ["fc1.weight","fc2.weight"]:
    diff=np.argwhere(V[t]!=W[t])
    for r_,c_ in diff: ops.append({"tensor":t,"rows":[int(r_)],"cols":[int(c_)],"op":"set","value":[[float(V[t][r_,c_])]]})
ops.append({"tensor":"fc2.bias","rows":list(range(p)),"cols":None,"op":"set","value":[float(v) for v in V["fc2.bias"]]})
print(len(ops)-1, fwd(V))
json.dump(ops,open("edit.json","w"))
