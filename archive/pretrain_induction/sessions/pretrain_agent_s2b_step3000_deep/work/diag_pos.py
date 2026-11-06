import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch, torch.nn.functional as F
W = T.base_weights(); P = W["pos.weight"]
Pc = P - P.mean(0)
Pn = F.normalize(Pc, dim=1)
C = Pn @ Pn.T
for lag in [1,2,3,4,8,16,32,64,128]:
    print(f"lag {lag:3d}: centered cos {torch.diagonal(C, lag)[64:].mean():.3f}")
print("pos row norm mean %.3f, centered %.3f, mean-vec %.3f" % (P.norm(dim=1).mean(), Pc.norm(dim=1).mean(), P.mean(0).norm()))
ev = torch.linalg.eigvalsh(Pc.T @ Pc).flip(0)
print("pos centered spectrum top-10 share", (ev[:10]/ev.sum()).tolist()[:10], "top64 %.3f" % (ev[:64].sum()/ev.sum()))
