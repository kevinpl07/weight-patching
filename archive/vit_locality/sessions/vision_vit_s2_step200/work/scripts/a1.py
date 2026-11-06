import sys; sys.path.insert(0, "../../../vision"); import agent_tools as T
import torch, json
W=T.base_weights()
m=T.metrics(edited=False)
print({k:m[k] for k in ['loss','acc','acc_aug','per_class_acc']})
for l in range(6): print(l, 'dist',m['dist'][l],'local',m['local'][l],'self',m['self'][l])
x,y=T.train_images(512,seed=1)
o=T.run(x,y,edited=False,resid=True)
print('loss',o['loss'].mean().item())
for l,r in enumerate(o['resid']):
    sd=r.std(-1).mean().item(); cv=r.var((0,1))
    print('resid',l,'tok std',round(sd,3),'coord var min/med/max',[round(v,4) for v in [cv.min().item(),cv.median().item(),cv.max().item()]])
pe=W['pos_embed'][0]
print('pos_embed norm per tok mean',pe.norm(dim=-1).mean().item(),'patch-embed resid std', o['resid'][0].std(-1).mean().item())
# pos similarity structure
pn=pe-pe.mean(0); pn=pn/pn.norm(dim=-1,keepdim=True); S=pn@pn.T
D=torch.load if False else None
from model import grid_dist
G=grid_dist(8)
for d in [1,2,3,5,7]:
    msk=(G-d).abs()<0.01
    print('pos cos sim at dist',d, round(S[msk].mean().item(),3))
for l in range(6):
    print('ln1 w',l, round(W[f'blocks.{l}.ln1.weight'].mean().item(),3), 'qk norm', round(W[f'blocks.{l}.attn.qkv.weight'][:384].norm().item(),3))
