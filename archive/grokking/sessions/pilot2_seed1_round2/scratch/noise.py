import numpy as np
exec(open("<project>/runs/seed1/loop/round2/agent/scratch/fair.py").read().split('h0=np.maximum')[0])
h=np.maximum(W1[:,A]+W1[:,p+B]+b1[:,None],0)
hc=np.maximum(V1[:,A]+V1[:,p+B]+b1[:,None],0)
rows=[]
tot_in=0; tot_out=0
for i in alive:
    full=np.outer(h[i]-h[i].mean(),W2[:,i])
    cl=np.outer(hc[i]-hc[i].mean(),V2[:,i])
    r_out=np.outer(h[i]-h[i].mean(),W2[:,i]-V2[:,i])   # output residual
    r_in=np.outer((h[i]-h[i].mean())-(hc[i]-hc[i].mean()),V2[:,i])  # input residual through clean output
    rows.append((i,dom[i],np.sqrt((cl**2).mean()),np.sqrt((r_in**2).mean()),np.sqrt((r_out**2).mean()),np.sqrt(((full-cl)**2).mean())))
rows.sort(key=lambda r:-r[5])
print("neuron freq clean_rms resid_in resid_out resid_total")
for r in rows: print("%3d %3d  %.3f  %.3f  %.3f  %.3f"%r)
