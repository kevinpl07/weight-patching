import torch, math
D=torch.arange(0,512).float()
for nf in (16,24,32):
  for lo,hi in [(2*math.pi/1024,math.pi*0.95),(2*math.pi/600,2.5),(2*math.pi/2000,2.0),(0.02,1.6)]:
    w=torch.exp(torch.linspace(math.log(lo),math.log(hi),nf))
    K=torch.cos(w[None]*(D[:,None]-1)).sum(1)
    other=torch.cat([K[:1],K[2:]])
    print(nf,round(lo,4),round(hi,2),"K1",K[1].item(),"K0",round(K[0].item(),2),"K2",round(K[2].item(),2),"max other",round(other.max().item(),2),"mean other",round(other[3:].mean().item(),2))
