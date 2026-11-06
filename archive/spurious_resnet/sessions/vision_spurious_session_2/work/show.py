from common_local import *
idx=[]
for k in DC:
    neg=torch.where(~clean&(y==k))[0]; idx.append(int(neg[0]))
T.show(idx,"work/spur_grid.png",ncol=10)
