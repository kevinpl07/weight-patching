import torch, numpy as np
D = torch.load("work/collect_base.pt")
gaps=[]; miss=0
for i,d in D.items():
    if not i.startswith("pns_") or not d["norm"]: continue
    ti, tl = d["topk_ids"][3].tolist(), d["topk_lp"][3].tolist()
    if 2411 in ti: gaps.append(tl[0]-tl[ti.index(2411)])
    else: miss+=1; gaps.append(tl[0]-tl[-1])
print("halluc ns: logprob gap top - ' .' (lower bound when missing):", np.round(np.percentile(gaps,[10,50,90,100]),2), "missing from top5:", miss)
print("nsp ns halluc median", np.median([d["nsp"] for i,d in D.items() if i.startswith("pns_") and d["norm"]]))
