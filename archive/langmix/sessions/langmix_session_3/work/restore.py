import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import torch
Wb = torch.load("work/lm_head_beta4.0.pt")
W0 = T.base_weights()['lm_head.weight']
print('delta fro', (Wb.float()-W0.float()).norm().item())
T.save_edit({"lm_head.weight": Wb}, note="FINAL: beta=4 Chinese-context-gated CJK logit penalty (PCA r=32, 279 dims, all 31,778 CJK rows)")
T.budget()
