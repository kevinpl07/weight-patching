import numpy as np, json, sys
th=float(sys.argv[1]); out=sys.argv[2]
frac,k=np.load("scratch/frac.npy")
rows=[int(i) for i in np.where(frac<th)[0]]
ops=[{"tensor":"fc1.bias","rows":rows,"op":"set","value":-10.0},
     {"tensor":"fc1.bias","rows":rows,"op":"reset_optimizer"}]
json.dump(ops,open(out,"w"))
print(len(rows),rows)
