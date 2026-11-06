"""Pretrain a small GPT on TinyStories with dense plateau probes.

Logs (out/log.jsonl):
  every step:   {"step", "loss", "lr"}
  every probe:  {"step", "probe": {...}} with
    val_loss            mean next-token loss, fixed story-aligned val batch
    loss_early/late     val loss at positions [8,24) / [160,256)  ->  icl = late - early (Olsson)
    rep1 / rep2         loss on 1st / 2nd copy of a repeated random-token sequence
                        (rep2 << rep1 once induction heads exist)
    prev[l][h]          previous-token attention score per head (val batch)
    ind[l][h]           induction attention score per head (repeated sequence:
                        query at t attends to t-L+1, the token after the earlier copy)

Data order depends only on --data-seed and the step number, so a run resumed from a
checkpoint (--init) sees exactly the same batches as the uninterrupted run.

Usage:
  .venv/Scripts/python.exe pretrain/train.py --out runs/pretrain/base_s0 --seed 0
  .venv/Scripts/python.exe pretrain/train.py --out runs/pretrain/arm_x --init <ckpt.pt>
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from model import GPT, GPTConfig

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DATA = os.path.join(HERE, "data")


def get_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--data-dir", default=DEFAULT_DATA, help="pretrain/data (TinyStories) or pretrain/data_fw (FineWeb-Edu)")
    ap.add_argument("--seed", type=int, default=0, help="init seed")
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--ctx", type=int, default=256)
    ap.add_argument("--attn-only", action="store_true")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--min-lr-frac", type=float, default=0.1)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--wd", type=float, default=0.1)
    ap.add_argument("--probe-every", type=int, default=25)
    ap.add_argument("--ckpt-every", type=int, default=250)
    ap.add_argument("--ckpt-steps", type=str, default="", help="extra comma-separated steps to checkpoint")
    ap.add_argument("--init", type=str, default="", help="resume model+optimizer+step from this checkpoint")
    ap.add_argument("--stop-at", type=int, default=0,
                    help="stop (and checkpoint) at this step; the LR schedule still targets --steps")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--mem-frac", type=float, default=0.85)
    return ap.parse_args()


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    p = min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup))
    return a.lr * (a.min_lr_frac + (1 - a.min_lr_frac) * 0.5 * (1 + math.cos(math.pi * p)))


class Data:
    def __init__(self, data_dir, ctx, batch, data_seed, device):
        self.train = np.memmap(os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r")
        self.val = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
        self.ctx, self.batch, self.seed, self.device = ctx, batch, data_seed, device

    def train_batch(self, step):
        rng = np.random.default_rng([self.seed, step])
        off = rng.integers(0, len(self.train) - self.ctx - 1, self.batch)
        x = np.stack([self.train[o:o + self.ctx + 1] for o in off]).astype(np.int64)
        x = torch.from_numpy(x).pin_memory().to(self.device, non_blocking=True)
        return x[:, :-1], x[:, 1:]

    def story_windows(self, n, length, seed):
        """n windows of `length` tokens, each starting at a story start and lying inside one story."""
        v = np.asarray(self.val)
        eos = np.flatnonzero(v == 0)
        starts = np.r_[0, eos[:-1] + 1]
        lens = eos - starts
        ok = starts[lens >= length]
        rng = np.random.default_rng(seed)
        pick = rng.choice(ok, n, replace=False)
        return torch.from_numpy(np.stack([v[o:o + length] for o in pick]).astype(np.int64)).to(self.device)

    def val_batch(self, n=128):
        x = self.story_windows(n, self.ctx + 1, 12345)  # whole window inside one long story
        return x[:, :-1], x[:, 1:]

    def text_rep_batch(self, n=64):
        """A real story prefix of L tokens, repeated twice (in-distribution induction probe)."""
        L = self.ctx // 2
        first = self.story_windows(n, L, 4242)
        return torch.cat([first, first], 1), L


def rep_batch(vocab, ctx, device, n=64, L=None):
    L = L or ctx // 2
    g = torch.Generator().manual_seed(777)
    first = torch.randint(1, vocab, (n, L), generator=g)
    return torch.cat([first, first], 1).to(device), L


def _rep_probe(model, rx, L):
    """Loss on 1st / 2nd copy of a repeated sequence, and per-head induction attention
    (query at t in the 2nd copy -> key t-L+1, the token after the earlier occurrence)."""
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, attn = model(rx[:, :-1], return_attn=True)
    rl = F.cross_entropy(logits.float().transpose(1, 2), rx[:, 1:], reduction="none")  # position t predicts rx[t+1]
    T = rx.shape[1] - 1
    q = torch.arange(L, T, device=rx.device)
    ind = [[round(v, 4) for v in a[:, :, q, q - L + 1].mean((0, 2)).tolist()] for a in attn]
    return rl[:, : L - 1].mean().item(), rl[:, L:].mean().item(), ind


@torch.no_grad()
def probe(model, vx, vy, tx, Lt, rx, Lr):
    model.eval()
    out = {}
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits, attn = model(vx, return_attn=True)
    tl = F.cross_entropy(logits.float().transpose(1, 2), vy, reduction="none")  # [B, T]
    pos = tl.mean(0)
    out["val_loss"] = tl.mean().item()
    out["loss_early"] = pos[8:24].mean().item()
    out["loss_late"] = pos[pos.shape[0] * 5 // 8:].mean().item()
    out["icl"] = out["loss_late"] - out["loss_early"]  # negative = later tokens benefit from context
    out["prev"] = [[round(v, 4) for v in a.diagonal(offset=-1, dim1=-2, dim2=-1).mean((0, 2)).tolist()] for a in attn]
    # repeated real story text (main induction probe)
    out["rep1"], out["rep2"], out["ind"] = _rep_probe(model, tx, Lt)
    # repeated uniform-random tokens (out of distribution; kept for reference)
    out["rrep1"], out["rrep2"], out["rind"] = _rep_probe(model, rx, Lr)
    model.train()
    return out


def save(path, model, opt, step, args):
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "step": step,
                "cfg": model.config_dict(), "args": vars(args)}, path)


def main():
    a = get_args()
    os.makedirs(os.path.join(a.out, "ckpt"), exist_ok=True)
    dev = "cuda"
    torch.cuda.set_per_process_memory_fraction(a.mem_frac)
    torch.backends.cuda.matmul.allow_tf32 = True
    meta = json.load(open(os.path.join(a.data_dir, "meta.json")))

    start = 0
    if a.init:
        ck = torch.load(a.init, map_location="cpu", weights_only=False)
        cfg = GPTConfig(**ck["cfg"])
        start = ck["step"]
    else:
        cfg = GPTConfig(vocab_size=meta["vocab_size"], ctx=a.ctx, n_layer=a.layers, n_head=a.heads,
                        d_model=a.d, attn_only=a.attn_only)
    torch.manual_seed(a.seed)
    model = GPT(cfg).to(dev)
    decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in model.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=True)
    if a.init:
        model.load_state_dict(ck["model"])
        if ck.get("opt") is not None:
            opt.load_state_dict(ck["opt"])
        del ck
    print(f"params {model.n_params()/1e6:.2f}M  cfg {cfg}  start step {start}", flush=True)
    fwd = torch.compile(model) if a.compile else model

    data = Data(a.data_dir, cfg.ctx, a.batch, a.data_seed, dev)
    pn = max(16, 128 * 256 // cfg.ctx // (2 if cfg.ctx > 256 else 1))  # probe batch: 128 at ctx 256, 32 at ctx 512
    vx, vy = data.val_batch(pn)
    tx, Lt = data.text_rep_batch(pn // 2)
    rx, Lr = rep_batch(cfg.vocab_size, cfg.ctx, dev, n=pn // 2)
    extra = {int(s) for s in a.ckpt_steps.split(",") if s}
    with open(os.path.join(a.out, "config.json"), "w") as f:
        json.dump({"args": vars(a), "cfg": model.config_dict(), "params": model.n_params()}, f, indent=2)

    log = open(os.path.join(a.out, "log.jsonl"), "a")
    t0 = time.time()
    end = a.stop_at or a.steps
    for step in range(start, end + 1):
        if step % a.probe_every == 0 or step == end:
            pr = probe(model, vx, vy, tx, Lt, rx, Lr)
            log.write(json.dumps({"step": step, "probe": pr}) + "\n")
            log.flush()
            print(f"step {step:5d}  val {pr['val_loss']:.3f}  icl {pr['icl']:+.3f}  rep1 {pr['rep1']:.2f} rep2 {pr['rep2']:.2f}"
                  f"  ind_max {max(max(r) for r in pr['ind']):.2f}  {time.time()-t0:.0f}s", flush=True)
        if (step % a.ckpt_every == 0 or step in extra) and step != start or (step == 0 and not a.init):
            save(os.path.join(a.out, "ckpt", f"step_{step:05d}.pt"), model, opt, step, a)
        if step == end:
            break
        lr = lr_at(step, a)
        for g in opt.param_groups:
            g["lr"] = lr
        x, y = data.train_batch(step)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = fwd(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        log.write(json.dumps({"step": step, "loss": round(loss.item(), 5), "lr": lr}) + "\n")
    save(os.path.join(a.out, "ckpt", f"step_{end:05d}.pt"), model, opt, end, a)
    print(f"done {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
