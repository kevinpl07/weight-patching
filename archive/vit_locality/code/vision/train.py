"""Train a small ViT on CIFAR-10 from scratch (AdamW + cosine, random crop + flip), with probes.

Logs (out/log.jsonl):
  every step:   {"step", "loss", "acc", "lr"}            (training batch, augmented)
  every probe:  {"step", "probe": {...}} with
    test_acc / test_loss     full CIFAR-10 test set (10,000 images, no augmentation)
    train_acc / train_loss   fixed 5,000 training images, no augmentation
    dist[l][h]               mean attention distance in patch units (500 fixed test images)
    local[l][h]              attention mass on the 3x3 neighbourhood incl. self
    self[l][h]               attention mass on the token itself

Batch order (and augmentation) depends only on --data-seed and the step number, so a run resumed with
--init sees exactly the same batches as the uninterrupted run. --stop-at N stops early while keeping the LR
schedule aimed at --steps. Checkpoints hold model + optimizer state + step + cfg + args.

Usage:
  .venv/Scripts/python.exe vision/train.py --out runs/vision_vit/base/s0 --seed 0 --stop-at 750 --ckpt-steps 750
  .venv/Scripts/python.exe vision/train.py --out <dir> --init <ckpt.pt>
"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data import Cifar  # noqa: E402
from model import ViT, ViTConfig, attn_stats  # noqa: E402


def get_args(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, default=0, help="init seed")
    ap.add_argument("--data-seed", type=int, default=None, help="batch-order seed (default: --seed)")
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--d", type=int, default=192)
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr-frac", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--probe-every", type=int, default=100)
    ap.add_argument("--ckpt-every", type=int, default=0)
    ap.add_argument("--ckpt-steps", type=str, default="")
    ap.add_argument("--init", type=str, default="", help="resume model+optimizer+step from this checkpoint")
    ap.add_argument("--stop-at", type=int, default=0)
    ap.add_argument("--mimetic0", action="store_true", help="mimetic attention init at step 0 (reference arm)")
    ap.add_argument("--local-sigma", type=str, default="", help="pilot only: fixed Gaussian locality bias, e.g. 1,2,4")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--mem-frac", type=float, default=0.2)
    a = ap.parse_args(argv)
    if a.data_seed is None:
        a.data_seed = a.seed
    return a


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    p = min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup))
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))


def make_opt(model, a):
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2 and n != "pos_embed"]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2 or n == "pos_embed"]
    return torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": nodecay, "weight_decay": 0.0}],
                             lr=a.lr, betas=(0.9, 0.999), fused=True)


@torch.no_grad()
def evaluate(model, x, y, bs=2500):
    tot_l, tot_c = 0.0, 0
    for i in range(0, len(x), bs):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lg = model(x[i:i + bs])
        tot_l += F.cross_entropy(lg.float(), y[i:i + bs], reduction="sum").item()
        tot_c += (lg.argmax(-1) == y[i:i + bs]).sum().item()
    return tot_l / len(x), tot_c / len(x)


@torch.no_grad()
def probe(model, data, xte, yte, trx, try_):
    model.eval()
    tl, ta = evaluate(model, xte, yte)
    rl, ra = evaluate(model, trx, try_)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        _, attns = model(xte[:500], return_attn=True)
    out = dict(test_acc=ta, test_loss=tl, train_acc=ra, train_loss=rl, **attn_stats(attns, model.cfg.grid))
    model.train()
    return out


def save(path, model, opt, step, a):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "cfg": model.config_dict(), "args": vars(a)}, path)


def main(argv=None):
    a = get_args(argv)
    os.makedirs(os.path.join(a.out, "ckpt"), exist_ok=True)
    dev = "cuda"
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    start = 0
    if a.init:
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        cfg = ViTConfig(**ck["cfg"])
        start = ck["step"]
        old = ck["args"]
        # the resumed run keeps the recipe of the run it continues
        for k in ("seed", "data_seed", "batch", "steps", "lr", "min_lr_frac", "warmup", "wd", "label_smoothing"):
            setattr(a, k, old[k])
    else:
        sig = [float(s) for s in a.local_sigma.split(",") if s]
        cfg = ViTConfig(d_model=a.d, n_layer=a.layers, n_head=a.heads, local_sigma=sig)
    torch.manual_seed(a.seed)
    model = ViT(cfg)
    if a.init:
        model.load_state_dict(ck["model"])
    elif a.mimetic0:
        from mimetic import mimetic_state
        sd = model.state_dict()
        sd.update(mimetic_state(sd, model.config_dict(), seed=a.seed))
        model.load_state_dict(sd)
    model = model.to(dev)
    opt = make_opt(model, a)
    if a.init and ck.get("opt") is not None:
        opt.load_state_dict(ck["opt"])
    if a.init:
        del ck
    print(f"params {model.n_params()/1e6:.3f}M  cfg {cfg}  start step {start}", flush=True)
    fwd = torch.compile(model) if a.compile else model

    data = Cifar(a.batch, a.data_seed, dev)
    xte, yte = data.test()
    g = torch.Generator().manual_seed(4321)
    tri = torch.randperm(data.n, generator=g)[:5000].to(dev)
    from data import normalize
    trx, try_ = normalize(data.xtr[tri]), data.ytr[tri]
    extra = {int(s) for s in a.ckpt_steps.split(",") if s}
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump({"args": vars(a), "cfg": model.config_dict(), "params": model.n_params(),
                   "steps_per_epoch": data.spe}, f, indent=2)

    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = time.time()
    end = a.stop_at or a.steps
    for step in range(start, end + 1):
        if step % a.probe_every == 0 or step == end:
            pr = probe(model, data, xte, yte, trx, try_)
            log.write(json.dumps({"step": step, "probe": pr}) + "\n")
            log.flush()
            md = [round(sum(r) / len(r), 2) for r in pr["dist"]]
            print(f"step {step:6d}  test {pr['test_acc']:.4f}  train {pr['train_acc']:.4f}  "
                  f"dist/layer {md}  {time.time()-t0:.0f}s", flush=True)
        if ((a.ckpt_every and step % a.ckpt_every == 0) or step in extra) and step != start:
            save(os.path.join(a.out, "ckpt", f"step_{step:05d}.pt"), model, opt, step, a)
        if step == end:
            break
        lr = lr_at(step, a)
        for gr in opt.param_groups:
            gr["lr"] = lr
        x, y = data.train_batch(step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = fwd(x)
        loss = F.cross_entropy(logits.float(), y, label_smoothing=a.label_smoothing)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        acc = (logits.argmax(-1) == y).float().mean()
        log.write(json.dumps({"step": step, "loss": round(loss.item(), 5), "acc": round(acc.item(), 4), "lr": lr}) + "\n")
    save(os.path.join(a.out, "ckpt", f"step_{end:05d}.pt"), model, opt, end, a)
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
