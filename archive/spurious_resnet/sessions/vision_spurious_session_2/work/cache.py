import sys; sys.path.insert(0, "../../../vision_spurious"); import agent_tools as T
import torch, time
t=time.time()
T.info()
d=T.dev()
F=[];L=[]
for i in range(0,len(d["labels"]),64):
    o=T.run(d["images"][i:i+64],edited=False,layers=["avgpool"])
    F.append(o["feats"]["avgpool"]);L.append(o["logits"])
F=torch.cat(F);L=torch.cat(L)
torch.save(dict(F=F,L=L),"work/base_feats.pt")
print(F.shape,time.time()-t)
