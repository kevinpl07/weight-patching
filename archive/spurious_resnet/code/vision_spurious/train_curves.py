"""Post-hoc fair training counterfactual (plan amendment A1): full training trajectories on the agents' dev data,
with checkpoints selected on dev only.

  python vision_spurious/train_curves.py cache          # held-out penultimate + layer3 features of the base model
  python vision_spurious/train_curves.py run [variants] # default: all five variants

Variants (all start from the pretrained weights; loss = CE on clean dev + mean -log(1 - p_k) on spurious dev):
  llr         fc (all rows), full-batch Adam lr 1e-4                      [same recipe as the original llr arm]
  llr_rows    only the 50 dev-class rows of fc (weight + bias), Adam lr 1e-4 (same scope as the agents)
  llr_gentle  fc, Adam lr 2e-5, + KL(original || new) on all 3,000 clean dev images
  ft          layer4 + fc, Adam lr 1e-5, batch 64, BN frozen              [same recipe as the original ft_l4 arm]
  ft_gentle   ft + KL(original || new) on the clean images of each batch
Selection, dev only:
  match_s{1,2,3}  first step whose in-sample dev spurious hit rate (as T.metrics computes it) <= the agent's in-sample
                  dev hit (s1 0.234, s2 0.165, s3 0.173)
  cv              3-fold within dev (hold out 1 of 3 clean photos per class and a third of each class's spurious
                  images): the step with the lowest CV hit among steps whose CV overall clean top-1 is >= CV base - 0.5
                  points and CV dev-class clean top-1 >= CV base - 3 points (the pass bar's damage limits)
Held-out eval (diagnostic curve + selected steps) never feeds back into selection.
"""
import importlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

EV = importlib.import_module("eval")
CACHE = C.RUNS / "cache"
OUT = C.RUNS / "curves"
TARGETS = dict(s1=0.234, s2=0.165, s3=0.173)
DEV = "cuda"


def cache():
    d = torch.load(C.DERIVED / "eval.pt")
    m = C.build()
    l3 = []
    h = m.layer3.register_forward_hook(lambda mod, i, o: l3.append(o.half().cpu()))
    _, phi = C.logits_feats(m, d["images"], feats=True, bs=32)
    h.remove()
    torch.save(dict(phi=phi, layer3=torch.cat(l3)), CACHE / "eval_feats.pt")
    print("cached eval", phi.shape)


def load():
    d = torch.load(C.DERIVED / "dev.pt")
    d.pop("images")
    d["clean"] = torch.tensor([k == "clean" for k in d["kind"]])
    c = torch.load(CACHE / "dev_feats.pt")
    e = torch.load(C.DERIVED / "eval.pt")
    e.pop("images")
    ec = torch.load(CACHE / "eval_feats.pt")
    return d, c, e, ec


def folds(d):
    """fold id 0..2 per dev image: clean = position among its class's 3 photos; spurious = position mod 3."""
    f = torch.zeros(len(d["labels"]), dtype=torch.long)
    cnt = {}
    for i, (y, c) in enumerate(zip(d["labels"].tolist(), d["clean"].tolist())):
        key = (y, c)
        f[i] = cnt.get(key, 0) % 3
        cnt[key] = cnt.get(key, 0) + 1
    return f


def dev_stats(logits, y, clean, dev_classes):
    """In-sample (or held-out-fold) stats as agent_tools.metrics computes them."""
    pred = logits.argmax(1)
    hits = [(pred[~clean & (y == k)] == k).float().mean().item() for k in dev_classes if (~clean & (y == k)).any()]
    dc = torch.isin(y, torch.tensor(dev_classes, device=y.device))
    return dict(hit=float(np.mean(hits)),
                top1=((pred == y) & clean).sum().item() / clean.sum().item(),
                top1_dc=((pred == y) & clean & dc).sum().item() / max(1, (clean & dc).sum().item()))


def loss_fn(z, y, clean, z0=None):
    lc = F.cross_entropy(z[clean], y[clean])
    p = z[~clean].softmax(1).gather(1, y[~clean, None]).squeeze(1)
    loss = lc - torch.log1p(-p.clamp(max=1 - 1e-6)).mean()
    if z0 is not None:
        loss = loss + F.kl_div(z[clean].log_softmax(1), z0[clean].log_softmax(1), log_target=True,
                               reduction="batchmean")
    return loss


class FC:
    """fc-only trainer on cached penultimate features (full batch)."""

    def __init__(self, sd, rows=None, lr=1e-4):
        self.W = torch.nn.Parameter(sd["fc.weight"].detach().clone().to(DEV))
        self.b = torch.nn.Parameter(sd["fc.bias"].detach().clone().to(DEV))
        self.mask = None
        if rows is not None:
            self.mask = torch.zeros(1000, device=DEV)
            self.mask[rows] = 1
        self.opt = torch.optim.Adam([self.W, self.b], lr=lr)

    def logits(self, phi):
        return phi @ self.W.T + self.b

    def step(self, phi, y, clean, z0):
        loss = loss_fn(self.logits(phi), y, clean, z0)
        self.opt.zero_grad()
        loss.backward()
        if self.mask is not None:
            self.W.grad *= self.mask[:, None]
            self.b.grad *= self.mask
        self.opt.step()

    def heldout_logits(self, ec):
        with torch.no_grad():
            return torch.cat([self.logits(ec["phi"][i:i + 4096].to(DEV)).cpu() for i in range(0, len(ec["phi"]), 4096)])

    def state(self):
        sd = C.base_state()
        sd["fc.weight"], sd["fc.bias"] = self.W.detach().cpu(), self.b.detach().cpu()
        return sd


class FT:
    """layer4 + fc trainer on cached layer3 outputs (minibatch 64, BN frozen)."""

    def __init__(self, lr=1e-5):
        self.m = C.build(device=DEV)
        for p in self.m.parameters():
            p.requires_grad_(False)
        ps = list(self.m.layer4.parameters()) + list(self.m.fc.parameters())
        for p in ps:
            p.requires_grad_(True)
        self.opt = torch.optim.Adam(ps, lr=lr)

    def logits(self, l3):
        return self.m.fc(torch.flatten(self.m.avgpool(self.m.layer4(l3.float())), 1))

    def step(self, l3, y, clean, z0):
        loss = loss_fn(self.logits(l3), y, clean, z0)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

    @torch.no_grad()
    def all_logits(self, l3, bs=256):
        return torch.cat([self.logits(l3[i:i + bs].to(DEV, non_blocking=True)).cpu() for i in range(0, len(l3), bs)])

    def heldout_logits(self, ec):
        return self.all_logits(ec["layer3"])

    def state(self):
        return {k: v.detach().cpu() for k, v in self.m.state_dict().items()}


SPEC = dict(
    llr=dict(kind="fc", lr=1e-4, kl=False, rows=False, steps=150),
    llr_rows=dict(kind="fc", lr=1e-4, kl=False, rows=True, steps=400),
    llr_gentle=dict(kind="fc", lr=2e-5, kl=True, rows=False, steps=800),
    ft=dict(kind="ft", lr=1e-5, kl=False, steps=270),
    ft_gentle=dict(kind="ft", lr=1e-5, kl=True, steps=270),
)


def train(name, d, c, sel, base_sd, z0_all, check_every, on_check, max_steps):
    """Train variant `name` on dev images where sel is True. Calls on_check(step, trainer, train_seconds)."""
    s = SPEC[name]
    y_all, clean_all = d["labels"], d["clean"]
    idx = sel.nonzero().squeeze(1)
    train_s = 0.0
    if s["kind"] == "fc":
        t = FC(base_sd, rows=d["dev_classes"] if s["rows"] else None, lr=s["lr"])
        phi, y, cl = c["phi"][idx].to(DEV), y_all[idx].to(DEV), clean_all[idx].to(DEV)
        z0 = z0_all[idx].to(DEV) if s["kl"] else None
        for step in range(max_steps + 1):
            if step % check_every(step) == 0:
                if on_check(step, t, train_s) is False:
                    break
            if step == max_steps:
                break
            t0 = time.time()
            t.step(phi, y, cl, z0)
            torch.cuda.synchronize()
            train_s += time.time() - t0
    else:
        t = FT(lr=s["lr"])
        g = torch.Generator().manual_seed(0)
        perm, pos = torch.randperm(len(idx), generator=g), 0
        for step in range(max_steps + 1):
            if step % check_every(step) == 0:
                if on_check(step, t, train_s) is False:
                    break
            if step == max_steps:
                break
            if pos + 64 > len(perm):
                perm, pos = torch.randperm(len(idx), generator=g), 0
            b = idx[perm[pos:pos + 64]]
            pos += 64
            t0 = time.time()
            t.step(c["layer3"][b].to(DEV), y_all[b].to(DEV), clean_all[b].to(DEV),
                   z0_all[b].to(DEV) if s["kl"] else None)
            torch.cuda.synchronize()
            train_s += time.time() - t0
    return t


def dev_logits(t, c, idx):
    with torch.no_grad():
        if isinstance(t, FC):
            return t.logits(c["phi"][idx].to(DEV)).cpu()
        return t.all_logits(c["layer3"][idx])


def run_variant(name, d, c, e, ec):
    s = SPEC[name]
    base_sd = C.base_state()
    z0_all = (c["phi"] @ base_sd["fc.weight"].T + base_sd["fc.bias"]).detach()
    y, clean, dcl = d["labels"], d["clean"], d["dev_classes"]
    fold = folds(d)
    every = (lambda st: 1) if s["kind"] == "fc" else (lambda st: 2 if st < 60 else 10)

    # 1) 3-fold CV within dev -> cv step
    cv = {}
    for f in range(3):
        tr, te = fold != f, (fold == f).nonzero().squeeze(1)

        def chk(step, t, _ts, f=f, te=te):
            cv.setdefault(step, []).append(dev_logits(t, c, te))
        train(name, d, c, tr, base_sd, z0_all, every, chk, s["steps"])
    te_all = torch.cat([(fold == f).nonzero().squeeze(1) for f in range(3)])
    cv_rows = []
    for step in sorted(cv):
        if len(cv[step]) < 3:
            continue
        st = dev_stats(torch.cat(cv[step]), y[te_all], clean[te_all], dcl)
        cv_rows.append(dict(step=step, **st))
    b0 = cv_rows[0]
    ok = [r for r in cv_rows if r["top1"] >= b0["top1"] - 0.005 and r["top1_dc"] >= b0["top1_dc"] - 0.03]
    cv_step = min(ok, key=lambda r: (r["hit"], r["step"]))["step"]

    # 2) full-dev trajectory: in-sample dev stats every check, held-out eval on a sparse grid + selected steps
    all_idx = torch.arange(len(y))
    pending = dict(TARGETS)
    selected, traj, curve = {}, [], []
    grid = set(range(0, 21, 2)) | set(range(20, 101, 5)) | set(range(100, 1001, 25))

    def heldout(t, step, label=None):
        L = t.heldout_logits(ec)
        res, _ = EV.score(L, e)
        if label:
            torch.save(L.half(), C.RUNS / "eval" / f"{name}_{label}_logits.pt")
            (C.RUNS / "eval" / f"{name}_{label}.json").write_text(json.dumps(dict(summary=dict(
                name=f"{name}_{label}", step=step, **res)), indent=1))
        return res

    def chk(step, t, ts):
        st = dev_stats(dev_logits(t, c, all_idx), y, clean, dcl)
        traj.append(dict(step=step, train_s=round(ts, 2), **st))
        labels = [k for k, v in list(pending.items()) if st["hit"] <= v]
        for k in labels:
            del pending[k]
        if step == cv_step:
            labels.append("cv")
        for lab in labels:
            res = heldout(t, step, lab if lab == "cv" else f"match_{lab}")
            selected[lab] = dict(step=step, train_s=round(ts, 2), dev=st, heldout=res)
            print(name, lab, step, round(st["hit"], 3), {k: round(res[k], 4) for k in
                  ("dev_spur_hit", "dev_spur_ap", "top1", "top1_dev_classes", "unseen_spur_hit")}, flush=True)
        if step in grid:
            curve.append(dict(step=step, **heldout(t, step)))
        if not pending and step >= cv_step and step >= 20 and st["hit"] <= 0.02:
            return False
    train(name, d, c, torch.ones(len(y), dtype=torch.bool), base_sd, z0_all, every, chk, s["steps"])
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(dict(spec=s, cv=cv_rows, cv_step=cv_step, selected=selected,
                                                      dev_traj=traj, heldout_curve=curve), indent=1))
    print(name, "done; cv_step", cv_step, "unreached targets", list(pending), flush=True)


def main():
    if sys.argv[1] == "cache":
        return cache()
    names = sys.argv[2:] or list(SPEC)
    d, c, e, ec = load()
    for n in names:
        t0 = time.time()
        run_variant(n, d, c, e, ec)
        print(n, "wall", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
