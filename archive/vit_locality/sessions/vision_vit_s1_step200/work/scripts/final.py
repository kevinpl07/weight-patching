import sys; sys.path.insert(0, "../../../vision"); import agent_tools as T
import torch
e = torch.load("work/C1x14.pt")
T.save_edit(e, note="FINAL C1x14: clean (row,col,r^2+c^2) code in pos_embed along content-null dirs (A=2) + 3 Q/K dims per head in blocks 0-2 giving Gaussian locality, sigma per head [2.1,3.5,5.6]")
B = T.base_weights()
for k,v in e.items(): print(k, "changed", int((v!=B[k]).sum()), "delta norm", round((v-B[k]).norm().item(),3), "base norm", round(B[k].norm().item(),3))
T.budget()
