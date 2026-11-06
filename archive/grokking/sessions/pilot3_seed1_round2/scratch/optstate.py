import numpy as np, sys
S=sys.argv[1]
O=dict(np.load(S+"/optimizer.npz"))
for k,v in O.items():
    if k.startswith("step"): print(k, v); continue
for t in ["fc1.weight","fc1.bias","fc2.weight","fc2.bias"]:
    m=O["exp_avg/"+t]; v=O["exp_avg_sq/"+t]; st=float(O["step/"+t])
    vh=v/(1-0.98**st); mh=m/(1-0.9**st)
    s=np.sqrt(vh)
    r=mh/(s+1e-8)
    print(t, "sqrt(vhat) median %.2e p10 %.2e p90 %.2e | |m|/sqrtv median %.3f mean %.3f"%(np.median(s),np.percentile(s,10),np.percentile(s,90),np.median(np.abs(r)),np.mean(np.abs(r))))
