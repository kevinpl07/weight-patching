"""Amendment A3: two more arms on cached penultimate features (no eval data in any selection).

  python vision_spurious/a3_arms.py script   # closed-form version of the agents' shared recipe (fixed variants)
  python vision_spurious/a3_arms.py dfr      # DFR-style balanced last-layer retraining, lambda grid, dev-only selection

script: per dev class k, with mu_s = mean feature of k's 25 dev cue images and mu_c = mean feature of k's clean photos
  (ImageNet pool, all ~50 photos of k; or the 3 ImageNetV2 dev photos in the *_dev3 variants):
  cue channels    = w_kj > 0 and mu_s_j >= r * mu_c_j, ranked by w_kj (mu_s_j - mu_c_j); top m scaled by (1 - a)
  object channels = w_kj > 0 and mu_c_j >= r * mu_s_j, ranked by w_kj (mu_c_j - mu_s_j); top m scaled by (1 + b)
  s2 settings: r=1.5, m=50, a=1.0, b=0.5 (session 2's submitted edit)
  s1 settings: r=2.0, m=32, a=0.75, b = (clean-k logit lost on the cue channels) / (sum over object channels of
               w_kj mu_c_j), capped at 1 (session 1's submitted edit)
dfr: start from the pretrained fc; full-batch Adam lr 1e-4; loss =
    CE(all pool-train photos + 3 dev photos per class)                                 [keeps the other classes stable]
  + weighted CE(pool-train photos of the 50 dev classes; weight 2 for photos whose cue score is above their class
    median, else 1; cue score = sum over k's s2 cue channels of w_kj phi_j)               [balanced positives, DFR-style]
  + mean -log(1 - p_k) over the cue images, averaged per class                         [balanced negatives]
  + lambda * (||W - W0||^2 + ||b - b0||^2)                                              [L2 toward the original fc]
  lambda in {0, 0.01, 0.1, 1}. Selection (dev only), as A2's practitioner rule: three runs each trained on 2/3 of the
  cue images; over all (lambda, step) pairs, the lowest CV cue hit among those with pool-val top-1 >= base - 0.5 points
  and pool-val dev-class top-1 >= base - 3 points. Matched points: for each agent target, the lambda whose first
  matched step has the highest pool-val dev-class top-1 (dev-only tie-break).
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import train_curves as TC  # noqa: E402

EV = TC.EV
DEV = "cuda"
OUTC = C.RUNS / "curves_in"
LAMS = [0.0, 0.01, 0.1, 1.0]


def load():
    meta = json.loads((C.DERIVED.parent / "imagenet_pool" / "meta.json").read_text())
    py = torch.tensor(meta["labels"])
    pval = torch.tensor([s == "val" for s in meta["split"]])
    pf = torch.load(C.RUNS / "cache" / "pool_feats.pt")
    d = torch.load(C.DERIVED / "dev.pt")
    d.pop("images")
    d["clean"] = torch.tensor([k == "clean" for k in d["kind"]])
    c = torch.load(C.RUNS / "cache" / "dev_feats.pt")
    c.pop("layer3")
    e = torch.load(C.DERIVED / "eval.pt")
    e.pop("images")
    ephi = torch.load(C.RUNS / "cache" / "eval_feats.pt")["phi"]
    return py, pval, pf["phi"], d, c["phi"], e, ephi


def score(W, b, ephi, e):
    L = ephi @ W.T + b
    return L, EV.score(L, e)[0]


def save_eval(name, L, res, extra=None):
    torch.save(L.half(), C.RUNS / "eval" / f"{name}_logits.pt")
    (C.RUNS / "eval" / f"{name}.json").write_text(json.dumps(dict(summary=dict(name=name, **res, **(extra or {}))), indent=1))


def dev_hit(W, b, dphi, d):
    z = dphi @ W.T + b
    pred = z.argmax(1)
    sp = ~d["clean"]
    return float(np.mean([(pred[sp & (d["labels"] == k)] == k).float().mean().item() for k in d["dev_classes"]]))


def channels(w, mu_s, mu_c, r, m):
    cue = (w > 0) & (mu_s >= r * mu_c)
    obj = (w > 0) & (mu_c >= r * mu_s)
    sc = torch.where(cue, w * (mu_s - mu_c), torch.tensor(-1e9))
    so = torch.where(obj, w * (mu_c - mu_s), torch.tensor(-1e9))
    ci = sc.argsort(descending=True)[:min(m, int(cue.sum()))]
    oi = so.argsort(descending=True)[:min(m, int(obj.sum()))]
    return ci, oi


def script():
    py, pval, pphi, d, dphi, e, ephi = load()
    sd = C.base_state()
    W0, b0 = sd["fc.weight"].detach().float(), sd["fc.bias"].detach().float()
    rows = []
    for name, r, m, a, bb in [("s2", 1.5, 50, 1.0, 0.5), ("s1", 2.0, 32, 0.75, None)]:
        for src in ("pool", "dev3"):
            W = W0.clone()
            for k in d["dev_classes"]:
                mu_s = dphi[~d["clean"] & (d["labels"] == k)].mean(0)
                mu_c = pphi[py == k].mean(0) if src == "pool" else dphi[d["clean"] & (d["labels"] == k)].mean(0)
                w = W0[k]
                ci, oi = channels(w, mu_s, mu_c, r, m)
                W[k, ci] = w[ci] * (1 - a)
                if bb is None:
                    lost = (a * w[ci] * mu_c[ci]).sum()
                    gain = (w[oi] * mu_c[oi]).sum()
                    beta = float(min(1.0, lost / gain)) if gain > 0 else 0.0
                else:
                    beta = bb
                W[k, oi] = w[oi] * (1 + beta)
            n = int((W != W0).sum())
            L, res = score(W, b0, ephi, e)
            nm = f"script_{name}_{src}"
            save_eval(nm, L, res, dict(entries_changed=n, dev_hit_in_sample=dev_hit(W, b0, dphi, d)))
            rows.append((nm, n, res))
            print(nm, n, {k: round(res[k], 4) for k in ("dev_spur_hit", "dev_spur_ap", "top1", "top1_dev_classes",
                                                          "unseen_spur_hit")}, flush=True)


def dfr():
    py, pval, pphi, d, dphi, e, ephi = load()
    sd = C.base_state()
    W0, b0 = sd["fc.weight"].detach().float().to(DEV), sd["fc.bias"].detach().float().to(DEV)
    dcl = d["dev_classes"]
    tr = ~pval
    # stability positives: all pool-train + dev clean
    Xs = torch.cat([pphi[tr], dphi[d["clean"]]]).to(DEV)
    Ys = torch.cat([py[tr], d["labels"][d["clean"]]]).to(DEV)
    # balanced dev-class positives with cue-based oversampling weights
    pos_i = (tr & torch.isin(py, torch.tensor(dcl))).nonzero().squeeze(1)
    wts = torch.ones(len(pos_i))
    for k in dcl:
        mu_s = dphi[~d["clean"] & (d["labels"] == k)].mean(0)
        mu_c = pphi[py == k].mean(0)
        ci, _ = channels(W0[k].cpu(), mu_s, mu_c, 1.5, 50)
        sel = (py[pos_i] == k).nonzero().squeeze(1)
        s = (pphi[pos_i[sel]][:, ci] * W0[k, ci].cpu()).sum(1)
        wts[sel] = torch.where(s > s.median(), 2.0, 1.0)
    Xp, Yp, Wp = pphi[pos_i].to(DEV), py[pos_i].to(DEV), (wts / wts.sum()).to(DEV)
    sp = (~d["clean"]).nonzero().squeeze(1)
    Xn_all, Yn_all = dphi[sp], d["labels"][sp]
    fold = TC.folds(d)[sp]
    # pool-val
    Xv, Yv = pphi[pval].to(DEV), py[pval].to(DEV)
    vdc = torch.isin(Yv.cpu(), torch.tensor(dcl)).to(DEV)
    every = lambda st: 1 if st < 50 else 5  # noqa: E731
    STEPS = 300

    def hit_of(z, y):
        p = z.argmax(1)
        return float(np.mean([(p[y == k] == k).float().mean().item() for k in dcl if (y == k).any()]))

    def train(lam, nsel, on_check):
        W = torch.nn.Parameter(W0.clone())
        b = torch.nn.Parameter(b0.clone())
        opt = torch.optim.Adam([W, b], lr=1e-4)
        Xn, Yn = Xn_all[nsel].to(DEV), Yn_all[nsel].to(DEV)
        ts = 0.0
        for st in range(STEPS + 1):
            if st % every(st) == 0:
                with torch.no_grad():
                    on_check(st, W.detach(), b.detach(), ts)
            if st == STEPS:
                break
            t0 = time.time()
            l1 = F.cross_entropy(Xs @ W.T + b, Ys)
            l2 = (F.cross_entropy(Xp @ W.T + b, Yp, reduction="none") * Wp).sum()
            pn = (Xn @ W.T + b).softmax(1).gather(1, Yn[:, None]).squeeze(1)
            l3 = -torch.log1p(-pn.clamp(max=1 - 1e-6)).mean()
            reg = lam * (((W - W0) ** 2).sum() + ((b - b0) ** 2).sum())
            loss = l1 + l2 + l3 + reg
            opt.zero_grad()
            loss.backward()
            opt.step()
            torch.cuda.synchronize()
            ts += time.time() - t0
        return W.detach(), b.detach()

    def val(W, b):
        p = (Xv @ W.T + b).argmax(1)
        return (p == Yv).float().mean().item(), (p == Yv)[vdc].float().mean().item()

    # practitioner rule over (lambda, step)
    cand, allrows = [], {}
    for lam in LAMS:
        rec = {}
        for f in range(3):
            def chk(st, W, b, ts, f=f):
                ho = fold == f
                z = (Xn_all[ho].to(DEV) @ W.T + b).cpu()
                va, vdc_ = val(W, b)
                rec.setdefault(st, []).append((z, Yn_all[ho], va, vdc_))
            train(lam, fold != f, chk)
        rows = []
        for st in sorted(rec):
            z = torch.cat([r[0] for r in rec[st]])
            y = torch.cat([r[1] for r in rec[st]])
            rows.append(dict(lam=lam, step=st, cv_hit=hit_of(z, y), val_top1=float(np.mean([r[2] for r in rec[st]])),
                             val_top1_dc=float(np.mean([r[3] for r in rec[st]]))))
        allrows[lam] = rows
        print("dfr lam", lam, "cv done", flush=True)
    b0r = allrows[LAMS[0]][0]
    ok = [r for lam in LAMS for r in allrows[lam]
          if r["val_top1"] >= b0r["val_top1"] - 0.005 and r["val_top1_dc"] >= b0r["val_top1_dc"] - 0.03]
    best = min(ok, key=lambda r: (r["cv_hit"], -r["step"]))
    print("dfr practitioner pick", best, flush=True)

    # full runs per lambda: trajectories, matched points, the picked (lambda, step)
    grid = set(range(0, 51, 5)) | set(range(50, 301, 25))
    match_cands = {s: [] for s in TC.TARGETS}
    for lam in LAMS:
        pending, selected, traj, curve = dict(TC.TARGETS), {}, [], []
        nm = f"dfr_lam{lam:g}"

        def chk(st, W, b, ts):
            h = dev_hit(W.cpu(), b.cpu(), dphi, d)
            va, vdc_ = val(W, b)
            traj.append(dict(step=st, train_s=round(ts, 2), dev_hit=h, val_top1=va, val_top1_dc=vdc_))
            labs = [k for k, v in list(pending.items()) if h <= v]
            for k in labs:
                del pending[k]
            labs = [f"match_{k}" for k in labs] + (["prac"] if (lam == best["lam"] and st == best["step"]) else [])
            for lab in labs:
                L, res = score(W.cpu(), b.cpu(), ephi, e)
                save_eval(f"{nm}_{lab}", L, res, dict(step=st))
                selected[lab] = dict(step=st, train_s=round(ts, 2), dev_hit=h, val_top1=va, val_top1_dc=vdc_,
                                     heldout=res)
                if lab.startswith("match_"):
                    match_cands[lab[6:]].append((vdc_, nm, lab))
            if st in grid:
                curve.append(dict(step=st, **score(W.cpu(), b.cpu(), ephi, e)[1]))
        train(lam, torch.ones(len(sp), dtype=torch.bool), chk)
        (OUTC / f"{nm}.json").write_text(json.dumps(dict(lam=lam, prac=allrows[lam], prac_pick=best, selected=selected,
                                                         dev_traj=traj, heldout_curve=curve, unreached=list(pending)),
                                                    indent=1))
        print(nm, "done; unreached", list(pending), flush=True)
    pick = {s: max(v)[1:] if v else None for s, v in match_cands.items()}
    (OUTC / "dfr_selection.json").write_text(json.dumps(dict(prac=best, match=pick), indent=1))
    print("dfr matched picks (by pool-val dev-class top-1):", pick)


if __name__ == "__main__":
    {"script": script, "dfr": dfr}[sys.argv[1]]()
