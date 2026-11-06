import torch, math
D=torch.arange(0,512).float()
for nf in (16,32):
  for lo,hi in [(2*math.pi/1024,math.pi*0.95),(0.05,2.5),(0.2,2.0),(0.3,1.5),(0.5,1.5)]:
   for lam in (0,0.5,1.0):
    w=torch.exp(torch.linspace(math.log(lo),math.log(hi),nf))
    K=(torch.cos(w[None]*(D[:,None]-1))-lam*torch.cos(w)[None]*torch.cos(w[None]*D[:,None])).sum(1)
    other=torch.cat([K[:1],K[2:]])
    print(nf,round(lo,3),round(hi,2),lam,"K1",round(K[1].item(),2),"K0",round(K[0].item(),2),"K2",round(K[2].item(),2),"max other",round(other.max().item(),2),"at",int(torch.cat([D[:1],D[2:]])[other.argmax()]),"gap/sqrt(nf)",round((K[1]-other.max()).item()/math.sqrt(nf),2))
