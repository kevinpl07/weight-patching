"""Scripted and fitted reference arms. All use the dev split only (data_vision/derived/dev.pt), never eval.pt.

  python vision_spurious/baselines.py feats     # cache base dev features (penultimate + layer3 output)
  python vision_spurious/baselines.py proj      # projection weight edit on fc rows of the 50 dev classes
  python vision_spurious/baselines.py spufix    # SpuFix-style clamp (inference-time wrapper, fc unchanged)
  python vision_spurious/baselines.py llr       # last-layer retraining (fc only)
  python vision_spurious/baselines.py ft        # short fine-tune of layer4 + fc (collateral-damage arm)

Direction used by proj and spufix, per dev class k: d_k = mean phi(spurious dev images of k) - mu0, where phi is the
penultimate feature and mu0 the mean phi over all 3,000 clean dev images (the "generic image").
Fitted arms (llr, ft) minimise  CE(clean dev images) + mean over spurious dev images of -log(1 - p_k(x)),
with Adam, starting from the pretrained weights, and stop when the dev spurious hit rate is <= 2% (or a step cap).
The stopping rule and hyperparameters were fixed in the plan before running; no tuning.
"""
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

ARMS = C.RUNS / "arms"
CACHE = C.RUNS / "cache" / "dev_feats.pt"


def load_dev():
    d = torch.load(C.DERIVED / "dev.pt")
    d["clean"] = torch.tensor([k == "clean" for k in d["kind"]])
    return d


def feats():
    d = load_dev()
    m = C.build()
    l3 = []
    h = m.layer3.register_forward_hook(lambda mod, i, o: l3.append(o.half().cpu()))
    _, phi = C.logits_feats(m, d["images"], feats=True, bs=32)
    h.remove()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(phi=phi, layer3=torch.cat(l3)), CACHE)
    print("cached", phi.shape, torch.cat(l3).shape)


def directions(d, phi):
    mu0 = phi[d["clean"]].mean(0)
    ks = d["dev_classes"]
    D = torch.stack([phi[~d["clean"] & (d["labels"] == k)].mean(0) - mu0 for k in ks])
    return ks, D, mu0


def proj():
    d, phi = load_dev(), torch.load(CACHE)["phi"]
    ks, D, mu0 = directions(d, phi)
    sd = C.base_state()
    W, b = sd["fc.weight"].clone(), sd["fc.bias"].clone()
    for k, dk in zip(ks, D):
        u = dk / dk.norm()
        c = W[k] @ u
        W[k] -= c * u
        b[k] += c * (u @ mu0)
    ARMS.mkdir(parents=True, exist_ok=True)
    torch.save({"fc.weight": W, "fc.bias": b}, ARMS / "proj_edit.pt")
    print("proj edit:", int((W != sd["fc.weight"]).sum() + (b != sd["fc.bias"]).sum()), "entries")


def spufix():
    d, phi = load_dev(), torch.load(CACHE)["phi"]
    ks, D, mu0 = directions(d, phi)
    W = C.base_state()["fc.weight"]
    V = W[ks] * D
    V = V / V.norm(dim=1, keepdim=True)
    ARMS.mkdir(parents=True, exist_ok=True)
    torch.save(dict(classes=ks, v=V, mu=W[ks] * mu0[None]), ARMS / "spufix_like.pt")
    print("spufix-like directions for", len(ks), "classes")


def loss_fn(logits, y, clean):
    lc = F.cross_entropy(logits[clean], y[clean])
    p = logits[~clean].softmax(1).gather(1, y[~clean, None]).squeeze(1)
    ls = -torch.log1p(-p.clamp(max=1 - 1e-6)).mean()
    return lc + ls, lc.item(), ls.item()


@torch.no_grad()
def dev_hit(logits, y, clean):
    return (logits[~clean].argmax(1) == y[~clean]).float().mean().item(), \
           (logits[clean].argmax(1) == y[clean]).float().mean().item()


def llr():
    d, phi = load_dev(), torch.load(CACHE)["phi"].cuda()
    y, clean = d["labels"].cuda(), d["clean"].cuda()
    sd = C.base_state()
    fc = torch.nn.Linear(2048, 1000).cuda()
    fc.load_state_dict({"weight": sd["fc.weight"], "bias": sd["fc.bias"]})
    opt = torch.optim.Adam(fc.parameters(), lr=1e-4)
    log = []
    for step in range(1001):
        z = fc(phi)
        hit, acc = dev_hit(z, y, clean)
        if step % 25 == 0 or hit <= 0.02:
            log.append(dict(step=step, dev_spur_hit=round(hit, 4), dev_clean_top1=round(acc, 4)))
            print(log[-1])
        if hit <= 0.02 or step == 1000:
            break
        loss, _, _ = loss_fn(z, y, clean)
        opt.zero_grad(); loss.backward(); opt.step()
    sd["fc.weight"], sd["fc.bias"] = fc.weight.detach().cpu(), fc.bias.detach().cpu()
    ARMS.mkdir(parents=True, exist_ok=True)
    torch.save(sd, ARMS / "llr_state.pt")
    (ARMS / "llr_log.json").write_text(json.dumps(log, indent=1))


def ft():
    d, cache = load_dev(), torch.load(CACHE)
    l3 = cache["layer3"]
    y, clean = d["labels"], d["clean"]
    m = C.build()
    for p in m.parameters():
        p.requires_grad_(False)
    params = list(m.layer4.parameters()) + list(m.fc.parameters())
    for p in params:
        p.requires_grad_(True)
    m.eval()  # BN keeps its running statistics
    opt = torch.optim.Adam(params, lr=1e-5)
    head = lambda x: m.fc(torch.flatten(m.avgpool(m.layer4(x)), 1))  # noqa: E731
    g = torch.Generator().manual_seed(0)
    log = []

    def dev_eval():
        with torch.no_grad():
            z = torch.cat([head(l3[i:i + 128].cuda().float()) for i in range(0, len(y), 128)])
        return dev_hit(z, y.cuda(), clean.cuda())

    for ep in range(11):
        hit, acc = dev_eval()
        log.append(dict(epoch=ep, dev_spur_hit=round(hit, 4), dev_clean_top1=round(acc, 4)))
        print(log[-1])
        if hit <= 0.02 or ep == 10:
            break
        perm = torch.randperm(len(y), generator=g)
        for i in range(0, len(y), 64):
            idx = perm[i:i + 64]
            z = head(l3[idx].cuda().float())
            loss, _, _ = loss_fn(z, y[idx].cuda(), clean[idx].cuda())
            opt.zero_grad(); loss.backward(); opt.step()
    sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    ARMS.mkdir(parents=True, exist_ok=True)
    torch.save(sd, ARMS / "ft_l4_state.pt")
    (ARMS / "ft_log.json").write_text(json.dumps(log, indent=1))


if __name__ == "__main__":
    {"feats": feats, "proj": proj, "spufix": spufix, "llr": llr, "ft": ft}[sys.argv[1]]()
