import sys; sys.path.insert(0, "../../../vision"); sys.path.insert(0, "work/scripts"); import agent_tools as T
import torch, math
from build import *
x,y = T.train_images(1024, seed=1)
o = T.run(x,y,edited=False,resid=True)
pw = W["patch_embed.weight"].reshape(D,-1)  # [192,48]
U,S,Vh = torch.linalg.svd(torch.cat([pw, W["patch_embed.bias"][:,None], torch.ones(D,1)],1), full_matrices=True)
rank = (S>1e-4*S[0]).sum().item(); print("content rank", rank)
N = U[:, rank:]  # null space of content (incl. ones)
print("null dim", N.shape[1])
C = 0
for l in (1,2):
    n = torch.nn.functional.layer_norm(o["resid"][l],(D,)).reshape(-1,D); n = n-n.mean(0)
    C = C + N.T@(n.T@n/len(n))@N
# also penalize being read by existing layer-0..2 weights (q,k,v, fc1)
Rd = 0
for l in (0,1,2):
    for nm in ("attn.qkv.weight","mlp.fc1.weight"):
        Wm = W[f"blocks.{l}.{nm}"]@N
        Rd = Rd + Wm.T@Wm
print("C trace/dim", (C.trace()/N.shape[1]).item(), "Rd trace/dim", (Rd.trace()/N.shape[1]).item())
M = C/ C.trace()*N.shape[1] + Rd/Rd.trace()*N.shape[1]
ev, V = torch.linalg.eigh(M)
dirs = N@V[:, :3]
print("chosen eig", ev[:3].tolist(), "median", ev.median().item())
torch.save(dirs, "work/dirs.pt")
for l in range(6):
    n = torch.nn.functional.layer_norm(o["resid"][l],(D,)).reshape(-1,D)
    print("layer",l,"std along dirs", [round(v,3) for v in (n@dirs).std(0).tolist()])
print("ones overlap", (dirs.sum(0)).tolist())
