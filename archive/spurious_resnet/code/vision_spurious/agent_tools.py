"""Tools for a weight-editing agent session on a pretrained ResNet-50 (torchvision IMAGENET1K_V1).

The session folder (AGENT_SESSION, default: current directory) contains session.json and work/ (notes, tool log, edit).

Usage from an agent script:
    import sys; sys.path.insert(0, "<repo>/vision_spurious"); import agent_tools as T
    T.info()                      # architecture summary, parameter count, budgets, the dev split
    W = T.base_weights()          # {name: CPU float32 tensor}, read-only copy of the weights
    D = T.dev()                   # dev images (uint8 [N,3,224,224]) and their labels / kind / ids
    names = T.categories()        # the 1000 ImageNet class names
    T.show(idx, "work/x.png")     # save a labelled grid of dev images to look at
    out = T.run(images, edited=True, layers=["layer4", "avgpool"])   # logits + activations
    m = T.metrics(edited=True)    # dev-split diagnostics (clean accuracy, spurious hit rate, AP)
    T.save_edit({"fc.weight": new}, note="..."); T.current_edit(); T.budget()

Forward passes are counted in images (budget AGENT_IMAGES, default 300,000). Every call is logged to
work/tool_log.jsonl. There are no gradient tools, and the rules forbid computing gradients yourself.
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SESSION = Path(os.environ.get("AGENT_SESSION", Path.cwd()))
WORK = SESSION / "work"
EDIT = WORK / "edited" / "edit.pt"
MAX_IMAGES = int(os.environ.get("AGENT_IMAGES", 300_000))
MAX_CHANGED = int(os.environ.get("AGENT_MAX_CHANGED", 200_000))
DEVICE = os.environ.get("AGENT_DEVICE", "cuda" if torch.cuda.is_available() else "cpu")
MAX_BATCH = 64
CHUNK = 16

_base = None
_dev = None
_models = {}


def _log(kind, **kw):
    WORK.mkdir(parents=True, exist_ok=True)
    with open(WORK / "tool_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(t=time.strftime("%H:%M:%S"), kind=kind, **kw)) + "\n")


def _spend(n):
    path = WORK / "budget.json"
    WORK.mkdir(parents=True, exist_ok=True)
    used = json.loads(path.read_text())["images"] if path.exists() else 0
    if used + n > MAX_IMAGES:
        raise RuntimeError(f"forward budget exhausted ({used:,}+{n} > {MAX_IMAGES:,} images)")
    path.write_text(json.dumps(dict(images=used + n, max=MAX_IMAGES)))


def budget():
    path = WORK / "budget.json"
    used = json.loads(path.read_text())["images"] if path.exists() else 0
    print(f"forward budget used {used:,}/{MAX_IMAGES:,} images; edit budget {MAX_CHANGED:,} entries; current edit: "
          f"{'yes' if EDIT.exists() else 'none'}")
    return used


def categories():
    """The 1000 ImageNet class names, index = class id. Free."""
    return list(C.CATEGORIES)


def base_weights():
    """The weights as CPU float32 tensors (BN num_batches_tracked counters omitted). Free. Copy before modifying."""
    global _base
    if _base is None:
        sd = C.base_state()
        _base = {k: sd[k].float().clone() for k in C.editable(sd)}
    return _base


def dev():
    """The dev split. Free. Returns a dict:
      images  uint8 [N, 3, 224, 224] (already resized/cropped; T.run normalizes them)
      labels  LongTensor [N]: for kind 'clean' the true ImageNet class; for kind 'spurious' the class k whose
              spurious cue the image shows (the image does NOT contain a class-k object)
      kind    list of 'clean' (ImageNetV2 photos, 3 per class) or 'spurious'
      ids     list of source file ids
      dev_classes  the classes that have spurious dev images"""
    global _dev
    if _dev is None:
        d = torch.load(C.DERIVED / "dev.pt")
        _dev = dict(images=d["images"], labels=d["labels"], kind=d["kind"], ids=d["ids"],
                    dev_classes=d["dev_classes"])
    return _dev


def info():
    """Print and return a summary of the model, budgets and dev split. Free."""
    W = base_weights()
    d = dev()
    n = {k: int(sum(1 for x in d["kind"] if x == k)) for k in ("clean", "spurious")}
    s = dict(model="torchvision resnet50, IMAGENET1K_V1 weights (eval mode; BN uses running stats)",
             params=sum(v.numel() for v in W.values()), n_tensors=len(W),
             edit_budget_entries=MAX_CHANGED, forward_budget_images=MAX_IMAGES,
             dev=dict(images=len(d["labels"]), **n, dev_classes={k: C.CATEGORIES[k] for k in d["dev_classes"]}))
    print(json.dumps(s, indent=1))
    return s


def show(idx, path, ncol=8, labels=None):
    """Save a grid of dev images (by index) as a PNG with a caption list in <path>.txt. Free."""
    from PIL import Image
    d = dev()
    idx = list(map(int, idx))[:64]
    ims = d["images"][idx]
    nrow = (len(idx) + ncol - 1) // ncol
    grid = torch.zeros(3, nrow * 226, ncol * 226, dtype=torch.uint8)
    for i, im in enumerate(ims):
        r, c = divmod(i, ncol)
        grid[:, r * 226:r * 226 + 224, c * 226:c * 226 + 224] = im
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(grid.permute(1, 2, 0).numpy()).save(path)
    cap = [f"{i}: idx {j} {d['kind'][j]} label {int(d['labels'][j])} ({C.CATEGORIES[int(d['labels'][j])]})"
           + (f" | {labels[i]}" if labels is not None else "") for i, j in enumerate(idx)]
    path.with_suffix(".txt").write_text("\n".join(cap))
    return str(path)


def current_edit():
    """The current edit as {name: full tensor}, or {} if none. Free."""
    return torch.load(EDIT, map_location="cpu") if EDIT.exists() else {}


def _get_model(edited):
    key = ("edited", EDIT.stat().st_mtime) if (edited and EDIT.exists()) else ("base",)
    if key not in _models:
        for k in [k for k in _models if k[0] == key[0]]:
            del _models[k]
        _models[key] = C.build(edit=current_edit() if key[0] == "edited" else None, device=DEVICE)
    return _models[key]


@torch.no_grad()
def run(images, edited=True, layers=()):
    """Forward pass on uint8 images [B, 3, 224, 224] (B <= 64; pass a dev index list or tensor slice).
    Returns a dict of CPU float32 tensors:
      logits [B, 1000]
      probs  [B, 1000] softmax
      feats  {layer: output}, for any module name in torchvision's resnet50, e.g. "layer3", "layer4.2",
             "layer4.2.conv3", "layer4.2.bn3", "avgpool" (penultimate, flattened to [B, 2048]). Module outputs are
             recorded as the module returns them (a Bottleneck output is after its final ReLU).
    Uses the current edit if edited=True and one exists. Counts B images against the forward budget."""
    images = torch.as_tensor(images)
    if images.dim() == 3:
        images = images[None]
    if images.dtype != torch.uint8:
        raise ValueError("pass uint8 images as stored in T.dev()['images']")
    B = images.shape[0]
    if B > MAX_BATCH:
        raise ValueError(f"{B} images > {MAX_BATCH} per call")
    _spend(B)
    m = _get_model(edited)
    mods = dict(m.named_modules())
    feats, hooks = {}, []
    for name in layers:
        if name not in mods:
            raise ValueError(f"unknown module {name}")
        hooks.append(mods[name].register_forward_hook(
            lambda mod, inp, out, name=name: feats.__setitem__(name, out.detach().float().cpu())))
    chunks, L = [], []
    try:
        for i in range(0, B, CHUNK):  # small chunks keep GPU memory low (the GPU is shared)
            feats.clear()
            L.append(m(C.prep(images[i:i + CHUNK], DEVICE)).float().cpu())
            chunks.append(dict(feats))
    finally:
        for h in hooks:
            h.remove()
    logits = torch.cat(L)
    feats = {k: torch.cat([c[k] for c in chunks]) for k in layers}
    if "avgpool" in feats:
        feats["avgpool"] = feats["avgpool"].flatten(1)
    _log("run", B=B, edited=edited and EDIT.exists(), layers=list(layers))
    return dict(logits=logits, probs=logits.softmax(1), feats=feats)


def _ap(y, s):
    o = np.argsort(-s, kind="stable")
    y = y[o]
    return float((np.cumsum(y) / np.arange(1, len(y) + 1) * y).sum() / y.sum())


def metrics(edited=True):
    """Diagnostics on the whole dev split (counts 4,250 images):
      clean_top1           top-1 on the 3,000 clean dev images
      clean_top1_devcls    top-1 on clean dev images of the 50 dev classes (150 images)
      spur_hit             mean over dev classes k of: fraction of k's spurious dev images predicted as k
      spur_pk              mean softmax prob of class k on k's spurious dev images
      spur_ap              mean over k of the average precision of p_k separating k's clean dev images (3) from
                           k's spurious dev images (25); 1.0 = every clean image ranks above every spurious one
      per_class            {k: dict(hit, pk, ap, clean_pk)}"""
    d = dev()
    L = []
    for i in range(0, len(d["labels"]), MAX_BATCH):
        L.append(run(d["images"][i:i + MAX_BATCH], edited=edited)["logits"])
    P = torch.cat(L).softmax(1)
    pred, y = P.argmax(1), d["labels"]
    clean = torch.tensor([k == "clean" for k in d["kind"]])
    dc = torch.tensor(d["dev_classes"])
    per = {}
    for k in d["dev_classes"]:
        pos, neg = clean & (y == k), ~clean & (y == k)
        s = torch.cat([P[pos, k], P[neg, k]]).numpy()
        lab = np.r_[np.ones(int(pos.sum())), np.zeros(int(neg.sum()))]
        per[k] = dict(hit=round((pred[neg] == k).float().mean().item(), 3), pk=round(P[neg, k].mean().item(), 3),
                      ap=round(_ap(lab, s), 3), clean_pk=round(P[pos, k].mean().item(), 3))
    res = dict(clean_top1=round(((pred == y) & clean).sum().item() / clean.sum().item(), 4),
               clean_top1_devcls=round(((pred == y) & clean & torch.isin(y, dc)).sum().item()
                                       / (clean & torch.isin(y, dc)).sum().item(), 4),
               spur_hit=round(float(np.mean([v["hit"] for v in per.values()])), 4),
               spur_pk=round(float(np.mean([v["pk"] for v in per.values()])), 4),
               spur_ap=round(float(np.mean([v["ap"] for v in per.values()])), 4),
               per_class=per)
    _log("metrics", edited=edited and EDIT.exists(), **{k: v for k, v in res.items() if k != "per_class"})
    return res


def save_edit(tensors, note=""):
    """Validate and save {param_name: full new tensor} as the current edit (replaces the previous one). Free.
    Rules: names must exist and shapes match; values must be finite; at most MAX_CHANGED entries may differ from the
    base weights in total."""
    base = base_weights()
    changed, clean, norms = 0, {}, {}
    for name, t in tensors.items():
        if name not in base:
            raise ValueError(f"unknown or out-of-scope tensor {name}")
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
