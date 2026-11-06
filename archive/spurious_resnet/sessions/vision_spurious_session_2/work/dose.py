from edit_fc2 import *
L0=F@Wf.T+bf; oth=clean&~torch.isin(y,torch.tensor(DC)); print("base nondev acc %.4f"%(L0[oth].argmax(1)==y[oth]).float().mean())
for kw in [dict(m=50,alpha=1.0,ratio=1.5,gamma=1.25),dict(m=50,alpha=1.0,ratio=1.5,gamma=1.5),dict(m=80,alpha=1.0,ratio=1.5,gamma=1.4),dict(m=50,alpha=0.75,ratio=1.5,gamma=1.25)]:
    Wn,bn,n=build(**kw); r,_=evalw(Wn,bn)
    print(kw,n,{k:round(float(v),3) for k,v in r.items()},"|",cv(**kw))
