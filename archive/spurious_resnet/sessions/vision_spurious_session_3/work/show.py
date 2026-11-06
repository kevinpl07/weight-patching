import sys; sys.path.insert(0, "../../../vision_spurious"); import agent_tools as T
import torch
d = T.dev(); y = d["labels"]; sp = [i for i,k in enumerate(d["kind"]) if k=="spurious"]
idx = []
for k in d["dev_classes"]:
    idx += [i for i in sp if int(y[i])==k][:1]
T.show(idx[:50], "work/spur_grid.png", ncol=10)
