"""Tools for a weight-editing agent session on a small pretraining checkpoint.

The session folder (AGENT_SESSION, default: current directory) contains:
    base.pt        the frozen checkpoint to edit (model weights only; the optimizer state stays with us)
    session.json   checkpoint path, data dir, model config, step
    loss_so_far.csv  training loss of this run up to the edit point
    work/          the agent's notes, scripts output, tool log and the current edit

Usage from an agent script:
    import sys; sys.path.insert(0, "<repo>/pretrain"); import agent_tools as T
    T.info()                           # model config, parameter shapes, training recipe
    W = T.base_weights()               # {name: CPU float32 tensor}, read-only copy of the checkpoint
    tok = T.tokenizer()                # tokenizers.Tokenizer (byte-level BPE, 8192 tokens); tok.decode(ids)
    x = T.train_windows(16, 512, seed=1)   # LongTensor [16, 512] of training text (Python code)
    out = T.run(x, edited=True, attn=True, resid=True)  # loss per position, attention, residual stream
    m = T.metrics(edited=True)         # loss on fixed training windows, copy benefit, per-head scores
    T.save_edit({"blocks.1.attn.qkv.weight": new}, note="...")
    T.current_edit(); T.budget()

Forward passes are cheap (milliseconds); they are counted, not tightly limited. Every call is logged to
work/tool_log.jsonl. There are no gradient tools, and the rules forbid computing gradients yourself.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from model import GPT, GPTConfig  # noqa: E402

SESSION = Path(os.environ.get("AGENT_SESSION", Path.cwd()))
WORK = SESSION / "work"
EDIT = WORK / "edited" / "edit.pt"
MAX_CALLS = int(os.environ.get("AGENT_CALLS", 1000))
MAX_CHANGED = int(os.environ.get("AGENT_MAX_CHANGED", 400_000))
OUT_OF_SCOPE = ("tok.weight", "head.weight")
MAX_TOKENS = 64 * 512  # per forward call

_sess = None
_base = None
_models = {}


def _session():
    global _sess
    if _sess is None:
        _sess = json.loads((SESSION / "session.json").read_text())
    return _sess


def _log(kind, **kw):
    WORK.mkdir(parents=True, exist_ok=True)
    with open(WORK / "tool_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(t=time.strftime("%H:%M:%S"), kind=kind, **kw)) + "\n")


def _spend(kind):
    path = WORK / "budget.json"
    WORK.mkdir(parents=True, exist_ok=True)
    used = json.loads(path.read_text())["used"] if path.exists() else 0
    if used >= MAX_CALLS:
        raise RuntimeError(f"forward-call budget exhausted ({used}/{MAX_CALLS})")
    path.write_text(json.dumps(dict(used=used + 1, max=MAX_CALLS)))


def budget():
    path = WORK / "budget.json"
    used = json.loads(path.read_text())["used"] if path.exists() else 0
    print(f"forward calls used {used}/{MAX_CALLS}; edit budget {MAX_CHANGED:,} entries; current edit: "
          f"{'yes' if EDIT.exists() else 'none'}")
    return used


def info():
    """Print and return the model config, step, parameter shapes and training recipe. Free."""
    s = _session()
    shapes = {k: tuple(v.shape) for k, v in base_weights().items()}
    d = dict(step=s["step"], total_steps=s["total_steps"], cfg=s["cfg"], recipe=s["recipe"], params=shapes)
    print(json.dumps(d, indent=1))
    return d


def base_weights():
    """The checkpoint's weights as CPU float32 tensors. Free. Don't modify in place; copy first."""
    global _base
    if _base is None:
        ck = torch.load(SESSION / "base.pt", map_location="cpu", weights_only=False)
        _base = {k: v.float().clone() for k, v in ck["model"].items()}
    return _base


def tokenizer():
    from tokenizers import Tokenizer
    return Tokenizer.from_file(str(Path(_session()["data_dir"]) / "tokenizer.json"))


def train_windows(n, length=512, seed=0):
    """n random windows of `length` tokens from the TRAINING split (Python files joined by <|eos|>=0). Free."""
    if length > _session()["cfg"]["ctx"]:
        raise ValueError(f"length > ctx {_session()['cfg']['ctx']}")
    data = np.memmap(Path(_session()["data_dir"]) / "train.bin", dtype=np.uint16, mode="r")
    rng = np.random.default_rng([10_000 + seed])
    off = rng.integers(0, len(data) - length, n)
    return torch.from_numpy(np.stack([data[o:o + length] for o in off]).astype(np.int64))


def current_edit():
    """The current edit as {name: full tensor}, or {} if none. Free."""
    return torch.load(EDIT, map_location="cpu") if EDIT.exists() else {}


def _get_model(edited):
    key = ("edited", EDIT.stat().st_mtime) if (edited and EDIT.exists()) else ("base",)
    if key not in _models:
        for k in [k for k in _models if k[0] == key[0]]:
            del _models[k]
        torch.cuda.set_per_process_memory_fraction(0.4)
        m = GPT(GPTConfig(**_session()["cfg"]))
        sd = dict(base_weights())
        if key[0] == "edited":
            sd.update(current_edit())
        m.load_state_dict(sd)
        _models[key] = m.cuda().eval()
    return _models[key]


@torch.no_grad()
def run(tokens, edited=True, attn=False, resid=False):
    """Forward pass on a LongTensor [B, T] (T <= 512, B*T <= 32768). Returns a dict of CPU tensors:
      loss   [B, T-1]  next-token loss at each position
      logits [B, T, V] only if B*T <= 4096 (else omitted)
      attn   list per layer of [B, H, T, T] attention patterns      (if attn=True)
      resid  list: [embed, after block 0, after block 1, ...] each [B, T, d] (if resid=True)
      head_out list per layer of [B, H, T, d]: each head's contribution to the residual stream (if resid=True)
    Uses the current edit if edited=True and one exists. Counts as one forward call."""
    tokens = torch.as_tensor(tokens)
    if tokens.dim() == 1:
        tokens = tokens[None]
    B, T = tokens.shape
    if B * T > MAX_TOKENS:
        raise ValueError(f"{B * T} tokens > {MAX_TOKENS} per call")
    _spend("run")
    m = _get_model(edited)
    x = tokens.cuda()
    h = m.tok(x) + m.pos(torch.arange(T, device=x.device))
    out = {"attn": [], "resid": [h.cpu()], "head_out": []}
    for blk in m.blocks:
        a = blk.attn
        z = blk.ln1(h)
        q, k, v = a.qkv(z).view(B, T, 3, a.h, a.dh).permute(2, 0, 3, 1, 4)
        s = (q @ k.transpose(-1, -2)) / a.dh ** 0.5
        s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool, device=x.device).tril(), float("-inf"))
        p = s.softmax(-1)
        y = p @ v  # [B, H, T, dh]
        Wo = a.proj.weight.view(a.proj.weight.shape[0], a.h, a.dh)  # [d, H, dh]
        ho = torch.einsum("bhtk,dhk->bhtd", y, Wo)
        h = h + ho.sum(1)
        if not blk.attn_only:
            h = h + blk.out(F.gelu(blk.fc(blk.ln2(h))))
        if attn:
            out["attn"].append(p.cpu())
        if resid:
            out["resid"].append(h.cpu())
            out["head_out"].append(ho.cpu())
    logits = m.head(m.ln_f(h))
    out["loss"] = F.cross_entropy(logits[:, :-1].transpose(1, 2), x[:, 1:], reduction="none").cpu()
    if B * T <= 4096:
        out["logits"] = logits.cpu()
    if not attn:
        del out["attn"]
    if not resid:
        del out["resid"], out["head_out"]
    _log("run", B=B, T=T, edited=edited and EDIT.exists(), attn=attn, resid=resid)
    return out


def metrics(edited=True, n=32):
    """Standard diagnostics on fixed TRAINING-split windows (counts as 3 forward calls):
      loss          mean next-token loss on n windows of 512 tokens
      loss_by_pos   mean loss in position buckets [0,64), [64,128), [128,256), [256,512)
      copy_benefit  loss on the 1st copy minus loss on the 2nd copy, for a 256-token training passage
                    repeated twice (high once the model can copy text it has already seen)
      prev[l][h]    mean attention from each position to the previous position
      ind[l][h]     on the repeated passage, mean attention from a token in the 2nd copy to the token right
                    after its earlier occurrence"""
    x = train_windows(n, 512, seed=999)
    o = run(x, edited=edited, attn=True)
    pos = o["loss"].mean(0)
    rep = train_windows(n // 2, 256, seed=998)
    rx = torch.cat([rep, rep], 1)[:, :511]
    L = 256
    r = run(rx, edited=edited, attn=True)
    q = torch.arange(L, rx.shape[1])
    res = dict(
        loss=round(o["loss"].mean().item(), 4),
        loss_by_pos=[round(pos[a:b].mean().item(), 3) for a, b in [(0, 64), (64, 128), (128, 256), (256, 511)]],
        copy_benefit=round((r["loss"][:, :L - 1].mean() - r["loss"][:, L:].mean()).item(), 4),
        prev=[[round(v, 3) for v in a.diagonal(offset=-1, dim1=-2, dim2=-1).mean((0, 2)).tolist()] for a in o["attn"]],
        ind=[[round(v, 3) for v in a[:, :, q, q - L + 1].mean((0, 2)).tolist()] for a in r["attn"]],
    )
    _spend("metrics")
    _log("metrics", edited=edited and EDIT.exists(), **{k: v for k, v in res.items() if k in ("loss", "copy_benefit")})
    return res


MAX_LOOKAHEADS = int(os.environ.get("AGENT_LOOKAHEADS", 0))
MAX_LOOKAHEAD_STEPS = 1500
LOOKAHEAD_DATA_SEED = 4242  # a different batch order from the real continuation


@torch.no_grad()
def _quick_eval(m):
    x = train_windows(32, 512, seed=999).cuda()
    rep = train_windows(16, 256, seed=998)
    rx = torch.cat([rep, rep], 1)[:, :511].cuda()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lx = F.cross_entropy(m(x[:, :-1]).float().transpose(1, 2), x[:, 1:], reduction="none")
        lr_ = F.cross_entropy(m(rx[:, :-1]).float().transpose(1, 2), rx[:, 1:], reduction="none")
    return round(lx.mean().item(), 4), round((lr_[:, :255].mean() - lr_[:, 256:].mean()).item(), 4)


def lookahead(steps=500, edited=True, every=100, label=""):
    """Continue TRAINING from this checkpoint for `steps` steps (<= 1500) with the real recipe: the same optimizer
    state, the real learning-rate schedule from this step on, batch 64 x 512. The batch ORDER differs from the
    real continuation (a different data seed). With edited=True the current edit is applied first; edited=False
    gives the no-edit counterfactual. Every `every` steps it reports the loss on the fixed metrics() training
    windows and the copy benefit. Returns a list of dicts. Limited budget (see T.budget()); each run takes about
    1-2 minutes. Use it to check how an edit you built by hand trains, not to search over edit parameters."""
    import types
    from train import Data, lr_at
    path = WORK / "lookahead_budget.json"
    used = json.loads(path.read_text())["used"] if path.exists() else 0
    if used >= MAX_LOOKAHEADS:
        raise RuntimeError(f"lookahead budget exhausted ({used}/{MAX_LOOKAHEADS})")
    if steps > MAX_LOOKAHEAD_STEPS:
        raise ValueError(f"steps > {MAX_LOOKAHEAD_STEPS}")
    WORK.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(used=used + 1, max=MAX_LOOKAHEADS)))
    s = _session()
    ck = torch.load(HERE.parent / s["source_ckpt"], map_location="cpu", weights_only=False)
    a = types.SimpleNamespace(**ck["args"])
    torch.cuda.set_per_process_memory_fraction(0.75)
    m = GPT(GPTConfig(**s["cfg"]))
    sd = dict(ck["model"])
    if edited and EDIT.exists():
        sd.update({k: v.to(sd[k].dtype) for k, v in current_edit().items()})
    m.load_state_dict(sd)
    m = m.cuda().train()
    decay = [p for n, p in m.named_parameters() if p.dim() >= 2]
    nodecay = [p for n, p in m.named_parameters() if p.dim() < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.wd}, {"params": nodecay, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.95), fused=True)
    opt.load_state_dict(ck["opt"])
    data = Data(s["data_dir"], s["cfg"]["ctx"], a.batch, LOOKAHEAD_DATA_SEED, "cuda")
    t0, start, hist, recent = time.time(), ck["step"], [], []
    m.eval(); l0, c0 = _quick_eval(m); m.train()
    hist.append(dict(step=start, loss=l0, copy_benefit=c0))
    for st in range(start, start + steps):
        for g in opt.param_groups:
            g["lr"] = lr_at(st, a)
        x, y = data.train_batch(st)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = m(x)
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        recent.append(loss.item())
        if (st + 1 - start) % every == 0:
            m.eval(); l, c = _quick_eval(m); m.train()
            hist.append(dict(step=st + 1, loss=l, copy_benefit=c, batch_loss=round(float(np.mean(recent)), 4)))
            recent = []
    del m, opt
    torch.cuda.empty_cache()
    out = WORK / "lookahead" / f"{used + 1:02d}_{'edited' if edited and EDIT.exists() else 'noedit'}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(dict(label=label, edited=edited and EDIT.exists(), steps=steps, history=hist), indent=1))
    _log("lookahead", steps=steps, edited=edited and EDIT.exists(), label=label, final=hist[-1],
         seconds=round(time.time() - t0))
    print(f"[lookahead {used + 1}/{MAX_LOOKAHEADS}] {'edited' if edited and EDIT.exists() else 'no edit'}: "
          + "  ".join(f"{h['step']}:{h['loss']:.3f}" for h in hist))
    return hist


def save_edit(tensors, note=""):
    """Validate and save {param_name: full new tensor} as the current edit (replaces the previous one). Free.
    Rules: names must exist and shapes match; tok.weight and head.weight are out of scope; values must be finite;
    at most MAX_CHANGED entries may differ from the checkpoint in total."""
    base = base_weights()
    changed, clean, norms = 0, {}, {}
    for name, t in tensors.items():
        if name not in base:
            raise ValueError(f"unknown parameter {name}")
        if name in OUT_OF_SCOPE:
            raise ValueError(f"{name} is out of scope")
        t = torch.as_tensor(t).detach().to("cpu", torch.float32).contiguous()
        if t.shape != base[name].shape:
            raise ValueError(f"shape mismatch for {name}: {tuple(t.shape)} vs {tuple(base[name].shape)}")
        if not torch.isfinite(t).all():
            raise ValueError(f"non-finite values in {name}")
        n = int((t != base[name]).sum())
        if n:
            changed += n
            clean[name] = t
            norms[name] = round((t - base[name]).norm().item(), 4)
    if changed > MAX_CHANGED:
        raise ValueError(f"edit changes {changed:,} entries; the limit is {MAX_CHANGED:,}")
    EDIT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clean, EDIT)
    _log("save_edit", tensors=sorted(clean), changed=changed, delta_norms=norms, note=note)
    print(f"saved edit: {len(clean)} tensors, {changed:,} entries changed, delta norms {norms}")
    return changed
