import sys; sys.path.insert(0, "../../../vision_spurious"); import agent_tools as T
import torch
d = T.dev(); y = d["labels"]; DC = d["dev_classes"]
sp = [i for i,k in enumerate(d["kind"]) if k=="spurious"]
idx = []
for k in DC:
    ii = [i for i in sp if int(y[i])==k]
    idx.append(ii[0])
T.show(idx, "work/spur_a.png", ncol=10)
idx = []
for k in DC:
    ii = [i for i in sp if int(y[i])==k]
    idx.append(ii[1])
T.show(idx, "work/spur_b.png", ncol=10)
