import numpy as np
exec(open("scratch/subsets.py").read().split("print(\"full\"")[0])
print(np.round(np.sort(score),2))
for t in [0.08,0.1,0.11,0.12,0.13,0.14,0.15,0.17]:
    al=(score>=t).astype(float)
    print(t, int(al.sum()), "raw",fwd(A,B,b1,W2,b2,al), "proj",fwd(Ac,Bc,b1,W2c,b2,al))
# max preactivation of each neuron over all a,b (weights only, no data)
mx=(A.max(1)+B.max(1)+b1); print("max preact range", mx.min(), mx.max())
