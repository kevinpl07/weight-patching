import sys; sys.path.insert(0, "../../../vision"); sys.path.insert(0, "work/scripts"); import agent_tools as T
import torch, math, json
from build import *
dirs = torch.load("work/dirs.pt")
x,y = T.train_images(1024, seed=1)
def kappas(A):
    T.save_edit(build(A=A, layers=(), dirs=dirs, kappa={}), note=f"pos only A={A}")
    o = T.run(x,y,edited=True,resid=True)
    ks = {}
    feat = torch.stack([rr/3.5, cc/3.5, s_t],1)  # [64,3]
    for l in range(6):
        n = torch.nn.functional.layer_norm(o["resid"][l],(D,))
        p = (n@dirs)  # [B,64,3]
        k = (p.mean(0)*feat).sum()/(feat**2).sum()/A
        ks[l] = k.item()
    return ks, o["loss"].mean().item()
def evaluate(cfg, save_as=None):
    A = cfg["A"]
    ks, lpos = kappas(A)
    e = build(A=A, sigma=cfg["sigma"], layers=cfg["layers"], offsets=cfg.get("offsets"), heads=cfg.get("heads"), dirs=dirs, kappa=ks)
    n = T.save_edit(e, note=json.dumps(cfg))
    m = T.metrics(edited=True)
    if save_as: torch.save(e, f"work/{save_as}.pt")
    print(json.dumps(cfg), "kappa", {k: round(v,2) for k,v in ks.items()}, "pos-only loss(1024)", round(lpos,4))
    print("  loss", m["loss"], "acc", m["acc"], "acc_aug", m["acc_aug"])
    print("  dist", m["dist"][:4]); print("  local", m["local"][:4]); print("  self", m["self"][:3])
    return m
if __name__ == "__main__":
    cfgs = json.loads(sys.argv[1])
    ob = T.run(x,y,edited=False); print("base loss(1024)", ob["loss"].mean().item(), "acc", (ob["logits"].argmax(-1)==y).float().mean().item())
    for c in cfgs: evaluate(c, c.get("save"))
