"""Weight-reading and weight-writing tools for the editing agent. All edits go through here.

    python edit_tools.py SESSION info
    python edit_tools.py SESSION stats
    python edit_tools.py SESSION show fc1.weight --rows 0:8 --cols 0:20
    python edit_tools.py SESSION probe --n 20 [--hidden]       # or --pairs "3,5 10,20"
    python edit_tools.py SESSION edit edit.json
    python edit_tools.py SESSION diff
    python edit_tools.py SESSION reset
    python edit_tools.py SESSION submit --note "what and why" [--predict-next X --predict-end Y]
    python edit_tools.py SESSION optim [--neurons] [--save FILE]   # gradient, AdamW moments, next update
    python edit_tools.py SESSION trial --steps N                  # throwaway training run from working.npz

An edit file is a JSON list of ops applied in order to the working weights:
    {"tensor": "fc1.weight", "rows": [3, 7] | null, "cols": [0, 1, 2] | null,
     "op": "set" | "add" | "scale", "value": number or nested list shaped like the selected block}
For 1-D tensors (biases) only "rows" is used. null selects everything along that axis.
"op": "reset_optimizer" (no value) marks the selected entries to have their AdamW moments zeroed when
training resumes. It doesn't count toward the budget and only takes effect on entries that were changed.
Every call is appended to SESSION/tool_log.jsonl.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))
from grok import MLP, Config, Run, encode, set_weights, DEVICE  # noqa: E402

np.set_printoptions(precision=4, suppress=True, linewidth=200, threshold=6000)
MAX_TRIALS = 3
MAX_TRIAL_STEPS = 1000


class Session:
    def __init__(self, path):
        self.path = Path(path)
        self.meta = json.loads((self.path / "meta.json").read_text())
        self.cfg = Config(**self.meta["cfg"])
        self.orig = dict(np.load(self.path / "weights.npz"))
        self.work = dict(np.load(self.path / "working.npz"))
        data = np.load(self.path / "train_pairs.npy")
        self.pairs, self.labels = data[:, :2], data[:, 2]
        self.train_set = {(int(a), int(b)) for a, b in self.pairs}
        self.submitted = (self.path / "submission.json").exists()
        f = self.path / "optreset.npz"
        self.reset_mask = dict(np.load(f)) if f.exists() else {k: np.zeros(v.shape, bool) for k, v in self.orig.items()}

    def effective_reset(self, weights=None):
        """Entries whose AdamW moments get zeroed: marked for reset and actually changed."""
        weights = weights or self.work
        return {k: self.reset_mask[k] & (weights[k] != self.orig[k]) for k in self.orig}

    def model(self, weights=None):
        m = MLP(self.cfg.p, self.cfg.hidden).to(DEVICE)
        set_weights(m, weights or self.work)
        return m.eval()

    @torch.no_grad()
    def train_metrics(self, weights=None):
        x, y = encode(self.pairs, self.cfg.p)
        logits = self.model(weights)(x)
        return dict(train_loss=round(F.cross_entropy(logits, y).item(), 6),
                    train_acc=round((logits.argmax(1) == y).float().mean().item(), 4))

    def changed(self, weights=None):
        weights = weights or self.work
        return {k: int((weights[k] != self.orig[k]).sum()) for k in self.orig}


def parse_slice(s, n):
    if s is None:
        return slice(None)
    a, _, b = s.partition(":")
    return slice(int(a) if a else 0, int(b) if b else n) if _ else int(a)


def cmd_info(s, args):
    m, c = s.meta, s.cfg
    print(f"Task: (a + b) mod {c.p}. Input = concat(one_hot(a), one_hot(b)), {2 * c.p} dims "
          f"(a in columns 0..{c.p - 1}, b in columns {c.p}..{2 * c.p - 1}). Output = {c.p} logits.")
    print(f"Model: fc1 Linear({2 * c.p},{c.hidden}) -> ReLU -> fc2 Linear({c.hidden},{c.p}). Cross-entropy loss.")
    print("Tensors:", {k: list(v.shape) for k, v in s.orig.items()}, "total", m["total_weights"])
    print(f"Training: full-batch AdamW lr={c.lr} weight_decay={c.weight_decay} betas=({c.beta1},{c.beta2}), "
          f"{len(s.pairs)} training pairs ({c.train_frac:.0%} of all {c.p * c.p}). Paused at step {m['step']}.")
    print("After you submit, training resumes from your weights with the same optimizer state and settings.")
    print(f"Edit budget: {m['budget']} changed entries; used {sum(s.changed().values())}. Submitted: {s.submitted}")
    print("History (step, train_loss, train_acc, held-out acc):")
    for h in m["history"] + [m["state"]]:
        print(f"  {h['step']:6d}  {h['train_loss']:.5f}  {h['train_acc']:.3f}  {h['test_acc']:.3f}")
    return f"budget used {sum(s.changed().values())}"


@torch.no_grad()
def cmd_stats(s, args):
    for k, v in s.work.items():
        print(f"{k:11s} shape {list(v.shape)} mean {v.mean():+.4f} std {v.std():.4f} "
              f"min {v.min():+.4f} max {v.max():+.4f} L2 {np.linalg.norm(v):.3f}")
    x, _ = encode(s.pairs, s.cfg.p)
    _, h = s.model()(x, return_hidden=True)
    h = h.cpu().numpy()
    w1, b1, w2 = s.work["fc1.weight"], s.work["fc1.bias"], s.work["fc2.weight"]
    p = s.cfg.p
    print("\nneuron | |w_in a| |w_in b| bias | |w_out| | active frac (train) | mean act")
    for i in range(w1.shape[0]):
        print(f"{i:6d} | {np.linalg.norm(w1[i, :p]):7.3f} {np.linalg.norm(w1[i, p:]):7.3f} {b1[i]:+.3f} | "
              f"{np.linalg.norm(w2[:, i]):7.3f} | {(h[:, i] > 0).mean():.3f} | {h[:, i].mean():.4f}")
    return "stats printed"


def cmd_show(s, args):
    w = s.work[args.tensor]
    rows = parse_slice(args.rows, w.shape[0])
    block = w[rows] if w.ndim == 1 else w[rows, parse_slice(args.cols, w.shape[1])]
    if block.size > 6000:
        raise SystemExit(f"block has {block.size} entries; show at most 6000 at a time")
    print(block)
    return f"showed {args.tensor} {args.rows} {args.cols}"


@torch.no_grad()
def cmd_probe(s, args):
    if args.pairs:
        pairs = np.array([[int(t) for t in tok.split(",")] for tok in args.pairs.split()])
        bad = [(int(a), int(b)) for a, b in pairs if (int(a), int(b)) not in s.train_set]
        if bad:
            raise SystemExit(f"not training pairs (held-out is off limits): {bad}")
    else:
        idx = np.random.default_rng(args.seed).choice(len(s.pairs), args.n, replace=False)
        pairs = s.pairs[idx]
    x, y = encode(pairs, s.cfg.p)
    logits, h = s.model()(x, return_hidden=True)
    prob = logits.softmax(1)
    for i, (a, b) in enumerate(pairs):
        line = (f"({a:2d},{b:2d}) label {y[i].item():2d} pred {logits[i].argmax().item():2d} "
                f"p(label) {prob[i, y[i]].item():.3f}")
        if args.hidden:
            top = torch.topk(h[i], 8)
            line += " | top hidden " + " ".join(f"{j}:{v:.2f}" for v, j in zip(top.values.tolist(), top.indices.tolist()))
        print(line)
    print("full train set:", s.train_metrics())
    return f"probed {len(pairs)} pairs"


def cmd_edit(s, args):
    if s.submitted:
        raise SystemExit("already submitted; no more edits")
    ops = json.loads(Path(args.file).read_text())
    new = {k: v.copy() for k, v in s.work.items()}
    mask = {k: v.copy() for k, v in s.reset_mask.items()}
    for op in ops:
        w = new[op["tensor"]]
        rows = np.arange(w.shape[0]) if op.get("rows") is None else np.asarray(op["rows"])
        if w.ndim == 1:
            idx = (rows,)
        else:
            cols = np.arange(w.shape[1]) if op.get("cols") is None else np.asarray(op["cols"])
            idx = np.ix_(rows, cols)
        if op["op"] == "reset_optimizer":
            mask[op["tensor"]][idx] = True
            continue
        val = np.asarray(op["value"], dtype=w.dtype)
        if op["op"] == "set":
            w[idx] = val
        elif op["op"] == "add":
            w[idx] += val
        elif op["op"] == "scale":
            w[idx] *= val
        else:
            raise SystemExit(f"unknown op {op['op']}")
    if not all(np.isfinite(v).all() for v in new.values()):
        raise SystemExit("edit rejected: non-finite values")
    used = sum(s.changed(new).values())
    if used > s.meta["budget"]:
        raise SystemExit(f"edit rejected: {used} changed entries > budget {s.meta['budget']}")
    before = s.train_metrics()
    np.savez(s.path / "working.npz", **new)
    np.savez(s.path / "optreset.npz", **mask)
    s.reset_mask = mask
    after = s.train_metrics(new)
    n_reset = sum(int(v.sum()) for v in s.effective_reset(new).values())
    print(f"applied {len(ops)} ops. budget used {used}/{s.meta['budget']}. optimizer resets on changed entries: "
          f"{n_reset}. train before {before} after {after}")
    return f"applied {len(ops)} ops, budget {used}, after {after}"


def cmd_diff(s, args):
    for k, n in s.changed().items():
        print(f"{k:11s} changed {n:5d}  delta L2 {np.linalg.norm(s.work[k] - s.orig[k]):.4f}")
    print(f"total changed {sum(s.changed().values())}/{s.meta['budget']}")
    return "diff"


def cmd_reset(s, args):
    if s.submitted:
        raise SystemExit("already submitted")
    np.savez(s.path / "working.npz", **s.orig)
    (s.path / "optreset.npz").unlink(missing_ok=True)
    print("working weights and optimizer-reset marks reset to the paused checkpoint")
    return "reset"


def cmd_submit(s, args):
    if s.submitted:
        raise SystemExit("already submitted")
    if s.meta.get("checkpoint") and (args.predict_next is None or args.predict_end is None):
        raise SystemExit("this session needs --predict-next and --predict-end (held-out accuracy as fractions 0-1)")
    np.savez(s.path / "submission.npz", **s.work)
    reset = s.effective_reset()
    np.savez(s.path / "submission_optreset.npz", **reset)
    info = dict(note=args.note, changed=sum(s.changed().values()), per_tensor=s.changed(),
                optimizer_resets={k: int(v.sum()) for k, v in reset.items()},
                predict_next=args.predict_next, predict_end=args.predict_end,
                train=s.train_metrics(), time=time.strftime("%Y-%m-%d %H:%M:%S"))
    (s.path / "submission.json").write_text(json.dumps(info, indent=1))
    print("submitted:", info)
    return f"submitted, changed {info['changed']}"


def optimizer_view(s):
    """Full-batch training gradient at the working weights, the AdamW moments (with marked resets applied),
    and the update the optimizer would make on its next step (decoupled weight decay included)."""
    f = s.path / "optimizer.npz"
    if not f.exists():
        raise SystemExit("no optimizer state in this session")
    o = np.load(f)
    c = s.cfg
    model = MLP(c.p, c.hidden).to(DEVICE)
    set_weights(model, s.work)
    x, y = encode(s.pairs, c.p)
    F.cross_entropy(model(x), y).backward()
    grads = {k: p.grad.detach().cpu().numpy() for k, p in model.named_parameters()}
    reset = s.effective_reset()
    out = {}
    for k, w in s.work.items():
        m, v = o[f"exp_avg/{k}"].copy(), o[f"exp_avg_sq/{k}"].copy()
        m[reset[k]], v[reset[k]] = 0, 0
        t = float(o[f"step/{k}"]) + 1
        g = grads[k]
        m1, v1 = c.beta1 * m + (1 - c.beta1) * g, c.beta2 * v + (1 - c.beta2) * g * g
        denom = np.sqrt(v1) / np.sqrt(1 - c.beta2 ** t) + 1e-8
        step = -c.lr * c.weight_decay * w - c.lr / (1 - c.beta1 ** t) * m1 / denom
        out.update({f"grad/{k}": g, f"exp_avg/{k}": m, f"exp_avg_sq/{k}": v, f"next_update/{k}": step.astype(w.dtype)})
    return out


def cmd_optim(s, args):
    o = optimizer_view(s)
    print("Full-batch training gradient at your working weights, AdamW moments (marked resets applied), and the\n"
          "next AdamW update (decoupled weight decay included). Updates are added to the weights.")
    for k, w in s.work.items():
        g, d = o[f"grad/{k}"], o[f"next_update/{k}"]
        cos = float((d * w).sum() / (np.linalg.norm(d) * np.linalg.norm(w) + 1e-12))
        print(f"{k:11s} |w| {np.linalg.norm(w):8.3f} |grad| {np.linalg.norm(g):9.5f} |next update| {np.linalg.norm(d):8.5f} "
              f"cos(update, w) {cos:+.3f}")
    if args.neurons:
        w1, w2 = s.work["fc1.weight"], s.work["fc2.weight"]
        g1, g2, gb = o["grad/fc1.weight"], o["grad/fc2.weight"], o["grad/fc1.bias"]
        d1, d2, db = o["next_update/fc1.weight"], o["next_update/fc2.weight"], o["next_update/fc1.bias"]
        print("\nneuron | |w_in| |grad_in| |upd_in| cos(upd_in,w_in) | bias grad bias upd | "
              "|w_out| |grad_out| |upd_out| cos(upd_out,w_out)")
        for i in range(w1.shape[0]):
            ci = (d1[i] @ w1[i]) / (np.linalg.norm(d1[i]) * np.linalg.norm(w1[i]) + 1e-12)
            co = (d2[:, i] @ w2[:, i]) / (np.linalg.norm(d2[:, i]) * np.linalg.norm(w2[:, i]) + 1e-12)
            print(f"{i:6d} | {np.linalg.norm(w1[i]):6.3f} {np.linalg.norm(g1[i]):8.5f} {np.linalg.norm(d1[i]):7.5f} {ci:+.3f} | "
                  f"{gb[i]:+.5f} {db[i]:+.5f} | {np.linalg.norm(w2[:, i]):6.3f} {np.linalg.norm(g2[:, i]):8.5f} "
                  f"{np.linalg.norm(d2[:, i]):7.5f} {co:+.3f}")
    if args.save:
        np.savez(args.save, **o)
        print(f"saved to {args.save} (keys: grad/, exp_avg/, exp_avg_sq/, next_update/ + tensor name)")
    return "optimizer view" + (f", saved {args.save}" if args.save else "")


def cmd_trial(s, args):
    if not s.meta.get("checkpoint"):
        raise SystemExit("trials are not available in this session")
    if s.submitted:
        raise SystemExit("already submitted")
    log = s.path / "trials.json"
    done = json.loads(log.read_text()) if log.exists() else []
    if len(done) >= MAX_TRIALS:
        raise SystemExit(f"trial budget used up ({MAX_TRIALS} per round)")
    if not 1 <= args.steps <= MAX_TRIAL_STEPS:
        raise SystemExit(f"--steps must be between 1 and {MAX_TRIAL_STEPS}")
    run = Run.load(s.meta["checkpoint"])
    set_weights(run.model, s.work)
    reset = s.effective_reset()
    with torch.no_grad():
        for k, p in run.model.named_parameters():
            mask = torch.as_tensor(reset[k], device=p.device)
            for key in ("exp_avg", "exp_avg_sq"):
                run.opt.state[p][key][mask] = 0
    start = run.step
    run.history = [run.evaluate()]
    run.train(args.steps)
    with torch.no_grad():
        _, h = run.model(run.xtr, return_hidden=True)
    firing = int((h > 0).any(0).sum())
    print(f"trial {len(done) + 1}/{MAX_TRIALS}: {args.steps} steps from your working weights (a throwaway copy; "
          f"optimizer state and settings as in the real run). Neurons firing on training pairs at the end: {firing}")
    print("steps | train loss | train acc | held-out acc")
    curve = []
    for m in run.history:
        print(f"{m['step'] - start:5d} | {m['train_loss']:.5f} | {m['train_acc']:.4f} | {m['test_acc']:.4f}")
        curve.append(dict(steps=m["step"] - start, train_loss=m["train_loss"], train_acc=m["train_acc"],
                          test_acc=m["test_acc"]))
    done.append(dict(steps=args.steps, changed=sum(s.changed().values()), firing_end=firing, curve=curve,
                     time=time.strftime("%H:%M:%S")))
    log.write_text(json.dumps(done, indent=1))
    last = curve[-1]
    return f"trial {len(done)}: {args.steps} steps, end train {last['train_acc']:.4f} held-out {last['test_acc']:.4f}"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("info")
    sub.add_parser("stats")
    p = sub.add_parser("show")
    p.add_argument("tensor")
    p.add_argument("--rows")
    p.add_argument("--cols")
    p = sub.add_parser("probe")
    p.add_argument("--pairs")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", action="store_true")
    p = sub.add_parser("edit")
    p.add_argument("file")
    sub.add_parser("diff")
    sub.add_parser("reset")
    p = sub.add_parser("submit")
    p.add_argument("--note", required=True)
    p.add_argument("--predict-next", type=float)
    p.add_argument("--predict-end", type=float)
    p = sub.add_parser("optim")
    p.add_argument("--neurons", action="store_true")
    p.add_argument("--save")
    p = sub.add_parser("trial")
    p.add_argument("--steps", type=int, required=True)
    args = ap.parse_args()

    session = Session(args.session)
    entry = dict(t=time.strftime("%H:%M:%S"), argv=sys.argv[2:])
    try:
        entry["result"] = globals()[f"cmd_{args.cmd}"](session, args)
        entry["ok"] = True
    except SystemExit as e:
        entry.update(ok=False, result=str(e))
        print(e)
    with open(session.path / "tool_log.jsonl", "a") as f:
        f.write(json.dumps(entry) + "\n")
