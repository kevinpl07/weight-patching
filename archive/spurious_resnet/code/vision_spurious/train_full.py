"""Amendment A2: the training counterfactual with a realistic amount of labelled data (ImageNet-train pool).

  python vision_spurious/train_full.py cache <pool_dir>             # base phi for the pool, layer3 for its val split
  python vision_spurious/train_full.py run <pool_dir> [variants]    # default: all four variants

Training data:
  positives = pool "train" split (~40 photos per class) + the 3 ImageNetV2 dev photos per class
  negatives = the 25 dev cue-only images per dev class (never any eval image)
Loss (A1 form, each term averaged over its own images, so positives and negatives get equal total weight per step):
  CE(positives) + mean -log(1 - p_k)(negatives) [+ KL(original || new) on the positives for *_gentle]
Variants:
  ft_gentle_in  layer4 + fc, Adam lr 1e-5, batches of 64 positives + 16 negatives, BN frozen, + KL
  ft_in         same without KL
  llr_gentle_in fc, full batch, Adam lr 2e-5, + KL
  llr_rows_in   only the 50 dev-class rows of fc (weight + bias), full batch, Adam lr 1e-4
Stopping (no eval data):
  match_sN  first step whose in-sample dev cue hit (as T.metrics computes it) <= agent sN's in-sample dev hit
  prac      practitioner rule: three runs, each trained on 2/3 of every class's cue images; at each check, the CV cue
            hit on the held-out thirds (pooled) and the pool-val top-1 (overall and on the 50 dev classes, averaged
            over the runs). Chosen step = lowest CV cue hit among steps with val top-1 >= base - 0.5 points and val
            dev-class top-1 >= base - 3 points. The full run (all 25 cue images) is then evaluated at that step.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import train_curves as TC  # noqa: E402

EV = TC.EV
DEV = "cuda"
import os
SMOKE = os.environ.get("A2_SMOKE")  # smoke test: tiny step counts, outputs under runs/vision_spurious/smoke_a2/
OUT = C.RUNS / ("smoke_a2" if SMOKE else "curves_in")
EVAL_OUT = OUT / "eval" if SMOKE else C.RUNS / "eval"
CACHE_F = (OUT if SMOKE else C.RUNS / "cache") / "pool_feats.pt"
SPEC = dict(
    ft_gentle_in=dict(kind="ft", lr=1e-5, kl=True, steps=400),
    ft_in=dict(kind="ft", lr=1e-5, kl=False, steps=400),
    llr_gentle_in=dict(kind="fc", lr=2e-5, kl=True, rows=False, steps=3000),
    llr_rows_in=dict(kind="fc", lr=1e-4, kl=False, rows=True, steps=1500),
)


def pool(pool_dir):
    meta = json.loads((Path(pool_dir) / "meta.json").read_text())
    arr = np.memmap(Path(pool_dir) / "images.u8", dtype=np.uint8, mode="r", shape=(meta["cap"] * 1000, 3, 224, 224))
    y = torch.tensor(meta["labels"])
    val = torch.tensor([s == "val" for s in meta["split"]])
    return arr, meta["N"], y, val


def u8(arr, idx):
    return torch.from_numpy(np.stack([arr[i] for i in idx]))


def cache(pool_dir):
    arr, N, y, val = pool(pool_dir)
    m = C.build()
    phi, l3v = [], []
    h = m.layer3.register_forward_hook(lambda mod, i, o: l3v.append(o.half().cpu()) if cache.keep else None)
    with torch.no_grad():
        for i in range(0, N, 64):
            idx = list(range(i, min(N, i + 64)))
            cache.keep = False
            x = C.prep(u8(arr, idx), DEV)
            f = C.backbone(m, x)
            phi.append(f.cpu())
        vidx = val.nonzero().squeeze(1).tolist()
        cache.keep = True
        for i in range(0, len(vidx), 64):
            C.backbone(m, C.prep(u8(arr, vidx[i:i + 64]), DEV))
    h.remove()
    torch.save(dict(phi=torch.cat(phi), val_l3=torch.cat(l3v), val_idx=torch.tensor(vidx)),
               CACHE_F)
    print("cached pool", N, "val", len(vidx))


cache.keep = False


class FTI(TC.FT):
    """layer4 + fc trainer that computes layer3 on the fly from raw images (frozen stem..layer3)."""

    @torch.no_grad()
    def l3(self, u):
        m = self.m
        x = m.maxpool(m.relu(m.bn1(m.conv1(C.prep(u, DEV)))))
        return m.layer3(m.layer2(m.layer1(x)))


def run_variant(name, P, d, c, e, ec):
    s = dict(SPEC[name])
    if SMOKE:
        s["steps"] = 12
    arr, N, py, pval, pf = P
    base_sd = C.base_state()
    W0, b0 = base_sd["fc.weight"].detach(), base_sd["fc.bias"].detach()
    tr_idx = (~pval[:N]).nonzero().squeeze(1)
    dclean = d["clean"].nonzero().squeeze(1)
    dspur = (~d["clean"]).nonzero().squeeze(1)
    # positives: pool train + dev clean. phi0 / labels per positive.
    pos_y = torch.cat([py[tr_idx], d["labels"][dclean]])
    pos_phi = torch.cat([pf["phi"][tr_idx], c["phi"][dclean]])
    pos_z0 = (pos_phi @ W0.T + b0)
    npos_pool = len(tr_idx)
    # validation (pool val split): labels + layer3 cache
    v_y = py[pf["val_idx"]]
    dcl = d["dev_classes"]
    v_dc = torch.isin(v_y, torch.tensor(dcl))
    # cue folds: position of each spurious image within its class, mod 3
    fold = TC.folds(d)[dspur]
    sp_y = d["labels"][dspur]
    every = (lambda st: 1 if st < 50 else 5) if s["kind"] == "fc" else (lambda st: 10 if st < 100 else 50)

    def val_stats(t):
        with torch.no_grad():
            if s["kind"] == "fc":
                z = t.logits(pf["phi"][pf["val_idx"]].to(DEV)).cpu()
            else:
                z = t.all_logits(pf["val_l3"])
        p = z.argmax(1)
        return (p == v_y).float().mean().item(), (p == v_y)[v_dc].float().mean().item()

    def spur_logits(t, sel):
        with torch.no_grad():
            if s["kind"] == "fc":
                return t.logits(c["phi"][dspur[sel]].to(DEV)).cpu()
            return t.all_logits(c["layer3"][dspur[sel]])

    def hit_of(z, yk):
        pred = z.argmax(1)
        return float(np.mean([(pred[yk == k] == k).float().mean().item() for k in dcl if (yk == k).any()]))

    def train(spur_sel, on_check):
        """spur_sel: bool mask over dspur used as negatives."""
        sp_idx = dspur[spur_sel]
        train_s = 0.0
        if s["kind"] == "fc":
            t = TC.FC(base_sd, rows=dcl if s["rows"] else None, lr=s["lr"])
            X = torch.cat([pos_phi, c["phi"][sp_idx]]).to(DEV)
            Y = torch.cat([pos_y, d["labels"][sp_idx]]).to(DEV)
            CL = torch.cat([torch.ones(len(pos_y), dtype=torch.bool), torch.zeros(len(sp_idx), dtype=torch.bool)]).to(DEV)
            Z0 = torch.cat([pos_z0, torch.zeros(len(sp_idx), 1000)]).to(DEV) if s["kl"] else None
            for step in range(s["steps"] + 1):
                if step % every(step) == 0 and on_check(step, t, train_s) is False:
                    break
                if step == s["steps"]:
                    break
                t0 = time.time()
                t.step(X, Y, CL, Z0)
                torch.cuda.synchronize()
                train_s += time.time() - t0
        else:
            t = FTI(lr=s["lr"])
            g = torch.Generator().manual_seed(0)
            dimg = d["images"]
            for step in range(s["steps"] + 1):
                if step % every(step) == 0 and on_check(step, t, train_s) is False:
                    break
                if step == s["steps"]:
                    break
                pi = torch.randint(0, len(pos_y), (64,), generator=g)
                si = sp_idx[torch.randint(0, len(sp_idx), (16,), generator=g)]
                t0 = time.time()
                a, b_ = pi[pi < npos_pool], pi[pi >= npos_pool] - npos_pool
                ims = [u8(arr, tr_idx[a].tolist())] if len(a) else []
                ims += [dimg[dclean[b_]]] if len(b_) else []
                ims.append(dimg[si])
                l3 = t.l3(torch.cat(ims))
                yb = torch.cat([pos_y[pi[pi < npos_pool]], pos_y[pi[pi >= npos_pool]], d["labels"][si]]).to(DEV)
                order_z0 = torch.cat([pos_z0[pi[pi < npos_pool]], pos_z0[pi[pi >= npos_pool]], torch.zeros(16, 1000)])
                clb = torch.cat([torch.ones(64, dtype=torch.bool), torch.zeros(16, dtype=torch.bool)]).to(DEV)
                t.step(l3, yb, clb, order_z0.to(DEV) if s["kl"] else None)
                torch.cuda.synchronize()
                train_s += time.time() - t0
        return t

    # 1) practitioner rule: 3 runs with 2/3 of the cue images, CV cue hit + pool-val damage
    rec = {}
    for f in range(3):
        def chk(step, t, ts, f=f):
            ho = fold == f
            z = spur_logits(t, ho)
            va, vdc = val_stats(t)
            rec.setdefault(step, []).append((z, sp_y[ho], va, vdc))
        train(fold != f, chk)
        print(name, "fold", f, "done", flush=True)
    prac_rows = []
    for step in sorted(rec):
        if len(rec[step]) < 3:
            continue
        z = torch.cat([r[0] for r in rec[step]])
        yk = torch.cat([r[1] for r in rec[step]])
        prac_rows.append(dict(step=step, cv_hit=hit_of(z, yk), val_top1=float(np.mean([r[2] for r in rec[step]])),
                              val_top1_dc=float(np.mean([r[3] for r in rec[step]]))))
    b = prac_rows[0]
    ok = [r for r in prac_rows if r["val_top1"] >= b["val_top1"] - 0.005 and r["val_top1_dc"] >= b["val_top1_dc"] - 0.03]
    prac_step = min(ok, key=lambda r: (r["cv_hit"], -r["step"]))["step"]

    # 2) full run with all 25 cue images per class
    pending, selected, traj, curve = dict(TC.TARGETS), {}, [], []
    grid = set(range(0, 101, 10)) | set(range(100, 3001, 50))

    def heldout(t, step, label=None):
        L = t.heldout_logits(ec)
        res, _ = EV.score(L, e)
        if label:
            torch.save(L.half(), EVAL_OUT / f"{name}_{label}_logits.pt")
            (EVAL_OUT / f"{name}_{label}.json").write_text(json.dumps(dict(summary=dict(
                name=f"{name}_{label}", step=step, **res)), indent=1))
        return res

    def chk(step, t, ts):
        h = hit_of(spur_logits(t, torch.ones(len(dspur), dtype=torch.bool)), sp_y)
        va, vdc = val_stats(t)
        traj.append(dict(step=step, train_s=round(ts, 2), dev_hit=h, val_top1=va, val_top1_dc=vdc))
        labels = [k for k, v in list(pending.items()) if h <= v]
        for k in labels:
            del pending[k]
        labels = [f"match_{k}" for k in labels] + (["prac"] if step == prac_step else [])
        for lab in labels:
            res = heldout(t, step, lab)
            selected[lab] = dict(step=step, train_s=round(ts, 2), dev_hit=h, val_top1=va, val_top1_dc=vdc, heldout=res)
            print(name, lab, step, round(h, 3), {k: round(res[k], 4) for k in
                  ("dev_spur_hit", "dev_spur_ap", "top1", "top1_dev_classes", "unseen_spur_hit")}, flush=True)
        if step in grid:
            curve.append(dict(step=step, **heldout(t, step)))
        if not pending and step >= prac_step and h <= 0.02:
            return False
    train(torch.ones(len(dspur), dtype=torch.bool), chk)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(dict(spec=s, prac=prac_rows, prac_step=prac_step, selected=selected,
                                                      dev_traj=traj, heldout_curve=curve,
                                                      unreached=list(pending)), indent=1))
    print(name, "done; prac_step", prac_step, "unreached targets", list(pending), flush=True)


def main():
    pool_dir = sys.argv[2]
    EVAL_OUT.mkdir(parents=True, exist_ok=True)
    CACHE_F.parent.mkdir(parents=True, exist_ok=True)
    if sys.argv[1] == "cache":
        return cache(pool_dir)
    arr, N, py, pval = pool(pool_dir)
    pf = torch.load(CACHE_F)
    d, c, e, ec = TC.load()
    d["images"] = torch.load(C.DERIVED / "dev.pt")["images"]
    for n in sys.argv[3:] or list(SPEC):
        t0 = time.time()
        run_variant(n, (arr, N, py, pval, pf), d, c, e, ec)
        print(n, "wall", round(time.time() - t0), "s", flush=True)


if __name__ == "__main__":
    main()
