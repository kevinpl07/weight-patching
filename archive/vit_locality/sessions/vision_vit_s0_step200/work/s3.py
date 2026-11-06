import sys; sys.path.insert(0,"work")
from build import *
d=torch.load('work/dirs.pt'); U,stds=d['U'],d['stds']
A={0:[1.0,2.0,None],1:[1.0,2.0,None],2:[1.0,2.0,None],3:[1.5,3.0,None]}
for dose in [0.25,0.5,2.0]:
    evaluate(f'A_dose{dose}', build(A,U=U,stds=stds,dose=dose), f'A dose {dose}')
C={0:[1.0,2.0,None],1:[1.0,2.0,None]}
evaluate('C_l01_h01', build(C,U=U,stds=stds), 'C')
