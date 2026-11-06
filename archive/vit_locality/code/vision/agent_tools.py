"""Tools for a weight-editing agent session on a CIFAR-10 ViT checkpoint in the middle of training.

The session folder (AGENT_SESSION, default: current directory) contains:
    base.pt          the frozen checkpoint to edit (model weights only; the optimizer state stays with us)
    session.json     step, model config, training recipe
    loss_so_far.csv  training loss and training-batch accuracy of this run up to the edit point
    work/            the agent's notes, script output, tool log and the current edit

Usage from an agent script:
    import sys; sys.path.insert(0, "<repo>/vision"); import agent_tools as T
    T.info()                              # model config, parameter shapes, training recipe
    W = T.base_weights()                  # {name: CPU float32 tensor}, read-only copy of the checkpoint
    x, y = T.train_images(256, seed=1)    # normalized float images [256,3,32,32] and labels, TRAINING set
    raw, y = T.train_set()                # the whole training set: uint8 [50000,32,32,3], labels [50000]
    out = T.run(x, edited=True, attn=True, resid=True)  # logits, per-image loss, attention, residual stream
    m = T.metrics(edited=True)            # loss/accuracy on fixed training images + per-head attention stats
    T.save_edit({"blocks.0.attn.qkv.weight": new}, note="...")
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
from model import ViT, ViTConfig, attn_stats, grid_dist  # noqa: E402
from data import normalize, augment, CLASSES  # noqa: E402

SESSION = Path(os.environ.get("AGENT_SESSION", Path.cwd()))
WORK = SESSION / "work"
EDIT = WORK / "edited" / "edit.pt"
MAX_CALLS = int(os.environ.get("AGENT_CALLS", 1000))
OUT_OF_SCOPE = ("patch_embed.weight", "patch_embed.bias", "head.weight", "head.bias")
MAX_IMAGES = 1024  # per forward call
MEM_FRAC = float(os.environ.get("AGENT_MEM_FRAC", 0.12))

_sess = None
_base = None
_train = None
_models = {}


def _session():
    global _sess
    if _sess is None:
        _sess = json.loads((SESSION / "session.json").read_text())
    return _sess


def _max_changed():
    return int(_session()["max_changed"])


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
    print(f"forward calls used {used}/{MAX_CALLS}; edit budget {_max_changed():,} entries; current edit: "
          f"{'yes' if EDIT.exists() else 'none'}")
    return used


def info():
    """Print and return the model config, step, parameter shapes and training recipe. Free."""
    s = _session()
    shapes = {k: tuple(v.shape) for k, v in base_weights().items()}
    d = dict(step=s["step"], total_steps=s["total_steps"], cfg=s["cfg"], recipe=s["recipe"],
             classes=list(CLASSES), params=shapes)
    print(json.dumps(d, indent=1))
    return d


def base_weights():
    """The checkpoint's weights as CPU float32 tensors. Free. Don't modify in place; copy first."""
    global _base
    if _base is None:
        ck = torch.load(SESSION / "base.pt", map_location="cpu", weights_only=False)
        _base = {k: v.float().clone() for k, v in ck["model"].items()}
    return _base


def train_set():
    """The whole CIFAR-10 TRAINING set: (uint8 numpy [50000,32,32,3] HWC RGB, int64 labels [50000]). Free.
    Class names: data.CLASSES / T.info()['classes']."""
    global _train
    if _train is None:
        d = np.load(_session()["train_npz"])
        _train = (d["x_train"], d["y_train"])
    return _train


def train_images(n, seed=0, augment_like_training=False):
    """n random TRAINING images, normalized exactly as in training: float [n,3,32,32] and labels [n]. Free.
    augment_like_training=True applies the training augmentation (random 4-pixel-padded crop, random flip)."""
    xs, ys = train_set()
    rng = np.random.default_rng([20_000 + seed])
    idx = rng.choice(len(xs), n, replace=False)
    x = normalize(torch.from_numpy(xs[idx]))
    if augment_like_training:
        oy, ox = torch.from_numpy(rng.integers(0, 9, n)), torch.from_numpy(rng.integers(0, 9, n))
        x = augment(x, oy, ox, torch.from_numpy(rng.random(n) < 0.5))
    return x, torch.from_numpy(ys[idx])


def current_edit():
    """The current edit as {name: full tensor}, or {} if none. Free."""
    return torch.load(EDIT, map_location="cpu") if EDIT.exists() else {}


def _get_model(edited):
    key = ("edited", EDIT.stat().st_mtime) if (edited and EDIT.exists()) else ("base",)
    if key not in _models:
        for k in [k for k in _models if k[0] == key[0]]:
            del _models[k]
        torch.cuda.set_per_process_memory_fraction(MEM_FRAC)
        m = ViT(ViTConfig(**_session()["cfg"]))
        sd = dict(base_weights())
        if key[0] == "edited":
            sd.update(current_edit())
        m.load_state_dict(sd)
        _models[key] = m.cuda().eval()
    return _models[key]


@torch.no_grad()
def run(images, labels=None, edited=True, attn=False, resid=False):
    """Forward pass (float32) on normalized images [B,3,32,32] (B <= 1024). Returns a dict of CPU tensors:
      logits [B,10];  loss [B] per-image cross-entropy (if labels given)
      attn     list per layer of [B,H,64,64] attention patterns (query x key; token t = grid row t//8, col t%8)
      resid    list [embed (patches + pos), after block 0, after block 1, ...], each [B,64,d]   (if resid=True)
      head_out list per layer of [B,H,64,d]: each head's contribution to the residual stream (incl. proj,
               excl. proj bias)                                                               (if resid=True)
      mlp_out  list per layer of [B,64,d]: each MLP's contribution                            (if resid=True)
    Uses the current edit if edited=True and one exists. Counts as one forward call."""
    x = torch.as_tensor(images).float()
    if x.dim() == 3:
        x = x[None]
    B = x.shape[0]
    if B > MAX_IMAGES:
        raise ValueError(f"{B} images > {MAX_IMAGES} per call")
    _spend("run")
    m = _get_model(edited)
    x = x.cuda()
    h = m.embed(x)
    out = {"attn": [], "resid": [h.cpu()], "head_out": [], "mlp_out": []}
    for blk in m.blocks:
        a = blk.attn
        z = blk.ln1(h)
        T_ = z.shape[1]
        q, k, v = a.qkv(z).view(B, T_, 3, a.h, a.dh).permute(2, 0, 3, 1, 4)
        p = ((q @ k.transpose(-1, -2)) / a.dh ** 0.5).softmax(-1)
        y = p @ v  # [B,H,T,dh]
        Wo = a.proj.weight.view(a.proj.weight.shape[0], a.h, a.dh)
        ho = torch.einsum("bhtk,dhk->bhtd", y, Wo)
        h = h + ho.sum(1) + a.proj.bias
        mo = blk.mlp(blk.ln2(h))
        h = h + mo
        if attn:
            out["attn"].append(p.cpu())
        if resid:
            out["resid"].append(h.cpu())
            out["head_out"].append(ho.cpu())
            out["mlp_out"].append(mo.cpu())
    logits = m.head(m.norm(h).mean(1))
    out["logits"] = logits.cpu()
    if labels is not None:
        out["loss"] = F.cross_entropy(logits, torch.as_tensor(labels).cuda(), reduction="none").cpu()
    if not attn:
        del out["attn"]
    if not resid:
        del out["resid"], out["head_out"], out["mlp_out"]
    _log("run", B=B, edited=edited and EDIT.exists(), attn=attn, resid=resid)
    return out


def metrics(edited=True):
    """Standard diagnostics on 4,096 fixed TRAINING images, no augmentation (counts as 2 forward calls):
      loss, acc        mean cross-entropy and accuracy
      acc_aug          accuracy on the same images with one fixed draw of the training augmentation
      per_class_acc    accuracy per class (order of info()['classes'])
      dist[l][h]       mean attention distance in patch units (attention-weighted Euclidean distance on the 8x8
                       grid, averaged over queries and images; uniform attention gives 4.14)
      local[l][h]      attention mass on the 3x3 neighbourhood of the query (incl. itself)
      self[l][h]       attention mass on the query token itself"""
    x, y = train_images(4096, seed=999)
    xa, _ = train_images(4096, seed=999, augment_like_training=True)
    m = _get_model(edited)
    with torch.no_grad():
        lg = torch.cat([m(x[i:i + 1024].cuda()).cpu() for i in range(0, 4096, 1024)])
        lga = torch.cat([m(xa[i:i + 1024].cuda()).cpu() for i in range(0, 4096, 1024)])
        _, attns = m(x[:512].cuda(), return_attn=True)
    _spend("metrics")
    _spend("metrics")
    pred = lg.argmax(-1)
    res = dict(loss=round(F.cross_entropy(lg, y).item(), 4), acc=round((pred == y).float().mean().item(), 4),
               acc_aug=round((lga.argmax(-1) == y).float().mean().item(), 4),
               per_class_acc=[round((pred[y == c] == c).float().mean().item(), 3) for c in range(10)],
               **attn_stats(attns, m.cfg.grid))
    _log("metrics", edited=edited and EDIT.exists(), loss=res["loss"], acc=res["acc"])
    return res


def save_edit(tensors, note=""):
    """Validate and save {param_name: full new tensor} as the current edit (replaces the previous one). Free.
    Rules: names must exist and shapes match; patch_embed.* and head.* are out of scope; values must be finite;
    at most session.json["max_changed"] entries may differ from the checkpoint in total."""
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
    if changed > _max_changed():
        raise ValueError(f"edit changes {changed:,} entries; the limit is {_max_changed():,}")
    EDIT.parent.mkdir(parents=True, exist_ok=True)
    torch.save(clean, EDIT)
    _log("save_edit", tensors=sorted(clean), changed=changed, delta_norms=norms, note=note)
    print(f"saved edit: {len(clean)} tensors, {changed:,} entries changed, delta norms {norms}")
    return changed
