import sys; sys.path.insert(0, "work/scripts")
from edit_lib import *
x,y=T.train_images(1024,seed=1)
o=T.run(x,y,edited=False,resid=True)
var=torch.stack([o['resid'][l].var((0,1))/o['resid'][l].var((0,1)).mean() for l in range(4)]).mean(0)
fcs=var.argsort()[:4].tolist(); print('feature coords',fcs, var[fcs])
s=[o['resid'][l].std(-1).mean().item() for l in range(6)]; print('s',s)
# also s per token spread
print('s cv layer0', (o['resid'][0].std(-1).std()/o['resid'][0].std(-1).mean()).item())
torch.save(dict(fcs=fcs,s=s),'work/calib.pt')
