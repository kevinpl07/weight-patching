import sys; sys.path.insert(0,"work")
from build import *
U,stds=directions()
print('stds',[round(s,3) for s in stds])
torch.save(dict(U=U,stds=stds),'work/dirs.pt')
specs={
 'A_l0-3_h01':{0:[1.0,2.0,None],1:[1.0,2.0,None],2:[1.0,2.0,None],3:[1.5,3.0,None]},
 'B_l0-2_all':{0:[1.0,1.5,2.5],1:[1.0,1.5,2.5],2:[1.0,2.0,3.0],3:[1.5,3.0,None]},
}
for n,sp in specs.items():
    evaluate(n, build(sp,U=U,stds=stds), n)
evaluate('posonly', build({},U=U,stds=stds),'pos only')
