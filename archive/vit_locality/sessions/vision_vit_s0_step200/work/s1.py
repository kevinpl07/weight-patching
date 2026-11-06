import sys; sys.path.insert(0, "../../../vision"); import agent_tools as T
import torch
from model import grid_dist
W=T.base_weights()
m=T.metrics(edited=False)
for k in ['loss','acc','acc_aug','per_class_acc']: print(k,m[k])
for l in range(6): print('L',l,'dist',m['dist'][l],'local',m['local'][l],'self',m['self'][l])
P=W['pos_embed'][0]
print('pos norm per token mean',P.norm(dim=1).mean().item())
Pc=P-P.mean(0)
C=torch.nn.functional.normalize(Pc,dim=1); S=C@C.T
D=grid_dist(8)
for d in [1,1.5,2,3,4,6]:
    msk=(D>d-0.3)&(D<d+0.3); print('pos cos sim at dist',d,round(S[msk].mean().item(),3))
x,y=T.train_images(512,seed=1)
o=T.run(x,y,edited=False,resid=True)
for i,r in enumerate(o['resid']): print('resid',i,'token std',r.std(-1).mean().item(),'norm',r.norm(dim=-1).mean().item())
# fraction of embed variance from pos
pe=o['resid'][0]
print('pos share of embed var', (Pc.pow(2).sum(1).mean()/ (pe-pe.mean(-1,keepdim=True)).pow(2).sum(-1).mean()).item())
for l in range(6): print('ln1 g mean',l,W[f'blocks.{l}.ln1.weight'].mean().item(), W[f'blocks.{l}.ln1.weight'].std().item())
for l in range(6):
  for h in range(3):
    print(l,h,'headout norm',o['head_out'][l][:,h].norm(dim=-1).mean().item(), 'mlp', o['mlp_out'][l].norm(dim=-1).mean().item() if h==0 else '')
