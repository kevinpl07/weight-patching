import json
ops=json.load(open("scratch/edit_flip.json"))
chosen={o["rows"][0] for o in ops if o["tensor"]=="fc1.weight"}|{o["cols"][0] for o in ops if o["tensor"]=="fc2.weight"}
noisy=[i for i in range(128) if i not in chosen]
print("noisy",len(noisy),"flip entries",len(ops))
ops.append({"tensor":"fc1.bias","rows":noisy,"op":"set","value":[0.0]*len(noisy)})
for t in ["fc1.weight","fc2.weight","fc1.bias"]:
    ops.append({"tensor":t,"rows":None,"cols":None,"op":"reset_optimizer"})
json.dump(ops,open("scratch/edit_v3.json","w"))
