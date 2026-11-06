"""Does a small code model memorize or generalize? Probes on a checkpoint.

1. train vs val loss on the same kind of windows (memorization -> train << val)
2. verbatim recall: given 128 tokens of a file, greedy-generate 32; exact-token match with the true continuation,
   for training files vs unseen validation files
3. novel identifiers: functions with made-up names; loss on the 1st vs later uses of each name
4. bracket closing: random nested (...[...{...}...]...) expressions with made-up names; does the correct closer
   get the highest probability among ) ] } ?
5. indentation: after `if <cond>:` / `def f(...):` / `for x in y:` lines, is the next line indented one level more?
6. a few greedy samples from hand-written prompts

Usage: .venv/Scripts/python.exe pretrain/probe_understanding.py runs/pretrain_code/gpt6L_d384_s0/ckpt/step_15000.pt
"""
import random
import string
import sys

import numpy as np
import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

sys.path.insert(0, "pretrain")
from model import GPT, GPTConfig  # noqa: E402

DATA = "pretrain/data_code"
dev = "cuda"


def load(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = GPT(GPTConfig(**ck["cfg"]))
    m.load_state_dict(ck["model"])
    return m.to(dev).eval(), ck["step"]


@torch.no_grad()
def logprobs(m, ids):
    x = torch.tensor(ids, device=dev)[None]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lg = m(x)
    return torch.log_softmax(lg.float()[0], -1)


@torch.no_grad()
def greedy(m, ids, n, ctx):
    ids = list(ids)
    for _ in range(n):
        lp = logprobs(m, ids[-ctx:])
        ids.append(int(lp[-1].argmax()))
    return ids


def doc_starts(arr, min_len):
    eos = np.flatnonzero(arr == 0)
    st = np.r_[0, eos[:-1] + 1]
    return st[(eos - st) >= min_len]


def windows_loss(m, arr, n, ctx, seed):
    rng = np.random.default_rng(seed)
    starts = doc_starts(arr[:20_000_000], ctx + 1)
    pick = rng.choice(starts, n, replace=False)
    x = torch.tensor(np.stack([arr[o:o + ctx + 1] for o in pick]).astype(np.int64), device=dev)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        lg = m(x[:, :-1])
    return F.cross_entropy(lg.float().transpose(1, 2), x[:, 1:]).item()


def recall(m, arr, n, seed, ctx, pre=128, gen=32):
    rng = np.random.default_rng(seed)
    starts = doc_starts(arr[:20_000_000], pre + gen + 300)
    pick = rng.choice(starts, n, replace=False)
    exact, frac = 0, []
    for o in pick:
        o = o + 200  # start mid-file, past license headers / imports
        p = arr[o:o + pre].astype(np.int64).tolist()
        truth = arr[o + pre:o + pre + gen].astype(np.int64).tolist()
        out = greedy(m, p, gen, ctx)[pre:]
        same = [a == b for a, b in zip(out, truth)]
        frac.append(np.mean(same))
        exact += all(same)
    return np.mean(frac), exact / n


def name(rng, k=None):
    k = k or rng.randint(5, 8)
    return "".join(rng.choice(string.ascii_lowercase) for _ in range(k))


def novel_identifiers(m, tok, rng, n=200):
    first, later = [], []
    for _ in range(n):
        f, a, b, c = name(rng), name(rng), name(rng), name(rng)
        src = (f"def {f}({a}, {b}):\n    {c} = {a} + {b}\n    if {c} > 0:\n        return {c}\n    return {a} - {b}\n\n\n"
               f"print({f}(1, 2))\n")
        enc = tok.encode(src)
        ids, offs = enc.ids, enc.offsets
        lp = logprobs(m, ids)
        tl = -lp[:-1].gather(-1, torch.tensor(ids[1:], device=dev)[:, None])[:, 0].cpu().numpy()
        for nm in (f, a, b, c):
            occ = [i for i in range(len(src)) if src.startswith(nm, i) and not src[i - 1].isalpha()]
            for j, ch in enumerate(occ):
                toks = [t for t, (s, e) in enumerate(offs) if s < ch + len(nm) and e > ch and t > 0]
                loss = tl[[t - 1 for t in toks]].sum()
                (first if j == 0 else later).append(loss)
    return np.mean(first), np.mean(later)


def brackets(m, tok, rng, n=300):
    pairs = {"(": ")", "[": "]", "{": "}"}
    correct, by_depth = 0, {}
    cands = {c: tok.encode(c).ids for c in ")]}"}
    for _ in range(n):
        depth = rng.randint(1, 4)
        opens = [rng.choice("([{") for _ in range(depth)]
        s = f"{name(rng)} = {name(rng)}"
        for o in opens:
            s += o
            s += {"(": f"{name(rng)}, ", "[": f"{rng.randint(0, 99)}, ", "{": f"'{name(rng, 4)}': "}[o]
        s += name(rng)
        want = pairs[opens[-1]]
        pre = tok.encode("x = 1\n" + s).ids
        lp = logprobs(m, pre)[-1]
        score = {c: lp[cands[c][0]].item() for c in cands}
        ok = max(score, key=score.get) == want
        correct += ok
        by_depth.setdefault(depth, []).append(ok)
    return correct / n, {d: round(float(np.mean(v)), 2) for d, v in sorted(by_depth.items())}


def indentation(m, tok, rng, n=200):
    ok = 0
    for _ in range(n):
        ind = " " * (4 * rng.randint(0, 2))
        head = rng.choice([f"if {name(rng)} > {rng.randint(0, 9)}:", f"def {name(rng)}({name(rng)}):",
                           f"for {name(rng)} in {name(rng)}:", f"while {name(rng)}:", f"class {name(rng).capitalize()}(object):"])
        # end the prompt at the colon: in training the newline and the next line's indentation are one token
        pre = f"{ind}{name(rng)} = {rng.randint(0, 99)}\n{ind}{head}"
        ids = tok.encode(pre).ids
        out = greedy(m, ids, 6, 512)
        text = tok.decode(out[len(ids):])
        line = text.split("\n")[1] if text.startswith("\n") else ""
        lead = len(line) - len(line.lstrip(" "))
        ok += lead == len(ind) + 4
    return ok / n


def main():
    path = sys.argv[1]
    m, step = load(path)
    tok = Tokenizer.from_file(f"{DATA}/tokenizer.json")
    ctx = m.cfg.ctx
    tr = np.memmap(f"{DATA}/train.bin", dtype=np.uint16, mode="r")
    va = np.memmap(f"{DATA}/val.bin", dtype=np.uint16, mode="r")
    rng = random.Random(0)
    print(f"== {path} (step {step}, {m.n_params()/1e6:.1f}M params)")
    print(f"1 loss          train {windows_loss(m, tr, 64, ctx, 1):.3f}   val {windows_loss(m, va, 64, ctx, 1):.3f}")
    ft, et = recall(m, tr, 100, 2, ctx)
    fv, ev = recall(m, va, 100, 2, ctx)
    print(f"2 recall        train files: {ft:.0%} tokens right, {et:.0%} exact 32-token continuations | "
          f"unseen files: {fv:.0%} tokens right, {ev:.0%} exact")
    f1, f2 = novel_identifiers(m, tok, rng)
    print(f"3 novel names   loss per name: first use {f1:.2f} nats, later uses {f2:.2f} nats")
    acc, bd = brackets(m, tok, rng)
    print(f"4 brackets      correct closer ranked first among ) ] }}: {acc:.0%}  by nesting depth {bd}  (chance 33%)")
    print(f"5 indentation   next line indented one level deeper: {indentation(m, tok, rng):.0%}")
    for p in ["def fibonacci(n):", "import os\n\n\ndef read_file(path):", "class Stack(object):\n    def __init__(self):"]:
        ids = tok.encode(p).ids
        print("6 sample  ---\n" + tok.decode(greedy(m, ids, 60, ctx)).rstrip() + "\n---")


if __name__ == "__main__":
    main()
