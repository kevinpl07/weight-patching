import sys; sys.path.insert(0, "../../../vision_spurious"); import agent_tools as T
import torch, numpy as np
d = T.dev(); W = T.base_weights(); names = T.categories()
c = torch.load("work/base_feats.pt"); F, L = c["F"], c["L"]
y = d["labels"]; clean = torch.tensor([k=="clean" for k in d["kind"]])
Wf, b = W["fc.weight"], W["fc.bias"]
P = L.softmax(1); pred = L.argmax(1)
mu_all = F[clean].mean(0); sd_all = F[clean].std(0)
print("clean top1", (pred[clean]==y[clean]).float().mean().item())
for k in d["dev_classes"]:
    sp = (~clean)&(y==k); cl = clean&(y==k)
    hit = (pred[sp]==k).float().mean().item()
    ms, mc = F[sp].mean(0), F[cl].mean(0)
    contrib = Wf[k]*(ms - mu_all)   # extra logit on spurious vs avg image
    contc = Wf[k]*(mc - mu_all)
    top = contrib.argsort(descending=True)[:5]
    # what do spurious get predicted as when not k
    others = pred[sp][pred[sp]!=k]
    oc = torch.bincount(others, minlength=1000).argsort(descending=True)[:2] if len(others) else []
    print(f"{k:4d} {names[k][:16]:16s} hit {hit:.2f} pk {P[sp,k].mean():.2f} cleanpk {P[cl,k].mean():.2f} "
          f"dSpur {contrib.sum():5.1f} dClean {contc.sum():5.1f} top5frac {contrib[top].sum()/contrib.sum():.2f} "
          f"top ch {top.tolist()} cl/sp on top {(contc[top].sum()/contrib[top].sum()):.2f} other-pred {[names[int(o)][:12] for o in oc]}")
