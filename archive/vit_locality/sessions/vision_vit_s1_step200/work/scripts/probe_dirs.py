import sys; sys.path.insert(0, "../../../vision"); sys.path.insert(0, "work/scripts"); import agent_tools as T
import torch, math
from build import *
x,y = T.train_images(1024, seed=1)
o = T.run(x,y,edited=False,resid=True)
dirs, ev = directions(o["resid"][0])
print("lowest content var eigen (normalized units)", ev.tolist())
torch.save(dirs, "work/dirs.pt")
# variance of those dirs in LN-normalized residual at each layer input, vs typical dir variance
for l in range(6):
    n = torch.nn.functional.layer_norm(o["resid"][l],(D,)).reshape(-1,D)
    p = n@dirs
    print("layer",l,"input: std along dirs", [round(v,3) for v in p.std(0).tolist()], "mean std per dim", round(n.std(0).mean().item(),3), "resid norm", round(o["resid"][l].norm(dim=-1).mean().item(),2))
