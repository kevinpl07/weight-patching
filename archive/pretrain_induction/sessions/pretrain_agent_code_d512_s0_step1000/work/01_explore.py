import sys; sys.path.insert(0, "../../../pretrain"); import agent_tools as T
import torch
T.info()
W=T.base_weights()
for k,v in W.items(): print(k, tuple(v.shape), round(v.norm().item(),3), round(v.std().item(),4))
m=T.metrics(edited=False)
print(m)
