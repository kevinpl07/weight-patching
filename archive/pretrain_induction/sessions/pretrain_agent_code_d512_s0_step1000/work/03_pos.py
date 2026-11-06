import sys; sys.path.insert(0, "../../../pretrain"); import agent_tools as T
import torch, torch.nn.functional as F
W=T.base_weights()
P=W["pos.weight"]; E=W["tok.weight"]
Pn=F.normalize(P,dim=-1); G=Pn@Pn.T
for d in [0,1,2,3,5,10,20,50,100]:
    print("cos lag",d, round(G.diagonal(-d).mean().item(),3))
print("P row norms", P.norm(dim=-1)[[0,1,2,5,10,50,100,300,511]])
print("E row norm mean", E.norm(dim=-1).mean().item())
S=torch.linalg.svdvals(P-P.mean(0)); print("P sv", (S[:20]/S[0]).numpy().round(3)); print("P energy top16/32/64",[(S[:k]**2).sum().item()/(S**2).sum().item() for k in (16,32,64)])
Se=torch.linalg.svdvals(E-E.mean(0)); print("E energy top64/128/256",[(Se[:k]**2).sum().item()/(Se**2).sum().item() for k in (64,128,256)])
# overlap of P subspace with E subspace
Up=torch.linalg.svd(P-P.mean(0),full_matrices=False)[2][:32]
Ce=(E-E.mean(0)).T@(E-E.mean(0))/E.shape[0]
print("E variance in top32 P dirs", (Up@Ce@Up.T).trace().item(), "total E var", Ce.trace().item())
