import sys; sys.path.insert(0,"work")
from build import *
d=torch.load('work/dirs.pt'); U,stds=d['U'],d['stds']
A={0:[1.0,2.0,None],1:[1.0,2.0,None],2:[1.0,2.0,None],3:[1.5,3.0,None]}
m=evaluate('FINAL_A_dose1', build(A,U=U,stds=stds,dose=1.0), 'FINAL: 2D locality bias, heads 0,1 of layers 0-3 (sigma 1/2, L3 1.5/3), coords written into pos_embed low-variance dirs')
print('per_class',m['per_class_acc'])
E=T.current_edit(); tot=sum(((E[k]-W[k])**2).sum() for k in E)**0.5
print('total delta norm',tot.item()); T.budget()
