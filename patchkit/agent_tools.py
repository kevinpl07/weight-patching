"""Budgeted tools for a blind weight-editing agent session on a patchkit task (see docs/AGENT_SESSIONS.md).

The agent never imports this file directly: `python -m patchkit.agent_session prepare` writes <session>/tools.py, a
shim that points AGENT_SESSION at the session folder and re-exports everything below. From an agent script:

    import sys; sys.path.insert(0, "<session>"); import tools as T
    T.info()                                  # model, tensor names/shapes, budgets, practice summary      (free)
    P = T.practice()                          # practice items: id, label, prompt, base reply, task fields  (free)
    T.prompt(0); T.tokenizer()                # exact chat-formatted prompt as scored; the tokenizer          (free)
    W = T.weight("model.layers.3.mlp.down_proj.weight", cols=[17])     # read (a slice of) a weight         (free)
    out = T.generate([0, 2, "Who is Ada Lovelace?"], edited=True)      # greedy replies           1 pass / 64 prompts
    A = T.activations([0, 2], layers=[10, 20], positions="last_prompt", kinds=("resid", "mlp_act"))   # same cost
    T.logit_lens([0, 2], layers=[10, 20], watch=["Sorry"])            # top next tokens per layer   same cost
    T.unit_usage([0, 2, 4], layer=20)                                 # per-neuron MLP activation stats   same cost
    T.set_cols("model.layers.20.mlp.down_proj.weight", [17], v)       # write values into the current edit (free)
    T.test(n=128)                                                     # task metrics on practice   1 pass / 64 items
    T.edit_summary(); T.save_copy("v1"); T.load_copy("v1"); T.clear_edit(); T.budget()             (free)

Every call is appended to <session>/work/tool_log.jsonl. Counted calls take the repo GPU lock (gpu_lock.py) and are
charged only once the GPU work starts. The model stays in CPU RAM for the life of the Python process and moves to the
GPU inside the lock for each counted call; the current edit is applied on the GPU copy and undone before it moves
back, so the CPU copy (and weight()) always holds the original weights. There are no gradient tools.

The edit is stored sparsely in work/edited/edit.safetensors: for each tensor `<name>::idx` (int64 flat indices into
the tensor) and `<name>::val` (the new bf16 values). Only entries whose new bf16 value differs from the original are
kept and counted.
"""
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import argparse  # noqa: E402
import contextlib  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

PK_ROOT = Path(__file__).resolve().parent.parent      # weight-patching/
REPO = PK_ROOT.parent                                 # project root (gpu_lock.py)
for _p in (str(PK_ROOT), str(REPO)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from patchkit import llm as L  # noqa: E402
from patchkit import task as TK  # noqa: E402

MAX_SEQS = 64                    # prompts / sequences per pass
MAX_NEW_CAP = 512                # max_new tokens per generate call
MAX_ACT_FLOATS = 250_000_000     # stored activation entries per activations() call (~1 GB float32)
OUT_OF_SCOPE = ("vision_tower", "multi_modal_projector")   # non-language parts of multimodal checkpoints
FWD_BATCH = 8                    # sequences per forward in teacher-forced tools

__all__ = ["info", "practice", "prompt", "tokenizer", "weight_names", "weight", "generate", "activations",
           "logit_lens", "unit_usage", "test", "set_rows", "set_cols", "set_entries", "set_tensor", "clear_edit",
           "current_edit", "edit_summary", "save_copy", "load_copy", "list_copies", "budget", "SESSION"]

SESSION = Path(os.environ.get("AGENT_SESSION", Path.cwd())).resolve()
WORK = SESSION / "work"
EDIT = WORK / "edited" / "edit.safetensors"
COPIES = WORK / "edits"

_S = dict(cfg=None, task=None, tok=None, model=None, practice=None, edit=None)


# ====================================================================== bookkeeping
def _cfg():
    if _S["cfg"] is None:
        p = SESSION / "session.json"
        if not p.exists():
            raise RuntimeError(f"{p} not found: AGENT_SESSION must point at a prepared session folder")
        _S["cfg"] = json.loads(p.read_text(encoding="utf-8"))
    return _S["cfg"]


def _log(kind, **kw):
    WORK.mkdir(parents=True, exist_ok=True)
    _clock()
    rec = dict(t=round(time.time(), 2), time=time.strftime("%H:%M:%S"), kind=kind, pid=os.getpid(), **kw)
    with open(WORK / "tool_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _clock():
    """Session clock: started by the first tool call. Returns minutes elapsed."""
    p = WORK / "clock.json"
    WORK.mkdir(parents=True, exist_ok=True)
    if not p.exists():
        p.write_text(json.dumps(dict(start=time.time(), start_str=time.strftime("%Y-%m-%d %H:%M:%S"))))
    return (time.time() - json.loads(p.read_text())["start"]) / 60


def _used():
    p = WORK / "budget.json"
    return json.loads(p.read_text())["used"] if p.exists() else 0


def _check(n, kind):
    mx = _cfg()["passes"]
    if _used() + n > mx:
        raise RuntimeError(f"forward-pass budget exhausted: {kind} needs {n} pass(es), {_used()}/{mx} used")


def _spend(n, kind):
    """Charge n passes. Called once the GPU lock is held, so a call killed while waiting is free."""
    _check(n, kind)
    used = _used() + n
    (WORK / "budget.json").write_text(json.dumps(dict(used=used, max=_cfg()["passes"])))
    print(f"[budget] passes {used}/{_cfg()['passes']} ({kind}); {_clock():.0f} of ~{_cfg()['minutes']} min",
          flush=True)


def budget():
    """Passes used, edit entries used, minutes elapsed. Free."""
    s = edit_summary(quiet=True)
    out = dict(passes_used=_used(), passes_max=_cfg()["passes"], entries_changed=s["entries_changed"],
               entries_max=_cfg()["max_changed"], minutes_elapsed=round(_clock(), 1), minutes=_cfg()["minutes"])
    print(json.dumps(out))
    _log("budget")
    return out


# ====================================================================== task, tokenizer, model
def _task():
    if _S["task"] is None:
        c = _cfg()
        t = TK.get(c["base_task"])
        t.model, t.name = c["model"], c["task"]
        _S["task"] = t
    return _S["task"]


def tokenizer():
    """The model's tokenizer (HF AutoTokenizer, left padding). Free."""
    if _S["tok"] is None:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(L.resolve(_cfg()["model"]))
        tok.padding_side = "left"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        _S["tok"] = tok
    return _S["tok"]


def _model():
    """The model on CPU (bf16, original weights). Same class choice as patchkit.llm.load, so names match the eval."""
    if _S["model"] is None:
        from transformers import AutoModelForCausalLM
        path = L.resolve(_cfg()["model"])
        t = time.time()
        try:
            m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16).eval()
        except (ValueError, KeyError):
            from transformers import AutoModelForImageTextToText
            m = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16).eval()
        m.requires_grad_(False)
        _S["model"] = m
        print(f"[tools] model loaded on CPU in {time.time() - t:.0f}s", flush=True)
    return _S["model"]


def _params():
    return dict(_model().named_parameters())


def _in_scope(name):
    return not any(s in name for s in OUT_OF_SCOPE)


# ====================================================================== free tools: data
def practice(label="any"):
    """The practice items (never held-out ones). Free. Each item is a dict with the task's own fields plus:
      id, prompt (the user message), label (1 = failure case to fix, 0 = must stay unchanged, None = unlabelled),
      base (the unedited model's greedy reply as scored).
    label: "any" (default), 1, 0 or None to filter."""
    if _S["practice"] is None:
        _S["practice"] = L.read_jsonl(SESSION / "practice" / "items.jsonl")
    xs = _S["practice"]
    if label != "any":
        xs = [x for x in xs if x["label"] == label]
    return xs


def _by_id():
    return {x["id"]: x for x in practice()}


def prompt(x):
    """The exact chat-formatted prompt string the model sees when scored (system prompt/tools of the task included).
    x: a practice id, a practice item, or a user message string. Free."""
    tok, t = tokenizer(), _task()
    if isinstance(x, int):
        x = _by_id()[x]
    if isinstance(x, dict):
        return t.render(tok, x)
    try:
        return t.render(tok, dict(prompt=x))
    except Exception:
        return L.chat(tok, x, t.base_system, tools=t.tools)


def _inputs(xs):
    """-> list of dict(key, prompt, reply). See generate() for the accepted forms."""
    if isinstance(xs, (int, str, dict)):
        xs = [xs]
    by = None
    out = []
    for x in xs:
        if isinstance(x, int):
            by = by or _by_id()
            if x not in by:
                raise KeyError(f"no practice item with id {x}")
            out.append(dict(key=x, prompt=prompt(by[x]), reply=by[x]["base"]))
        elif isinstance(x, str):
            out.append(dict(key=x[:60], prompt=prompt(x), reply=""))
        elif isinstance(x, dict) and "raw" in x:
            out.append(dict(key="raw:" + x["raw"][-40:], prompt=x["raw"], reply=""))
        elif isinstance(x, dict) and "id" in x:
            by = by or _by_id()
            out.append(dict(key=x["id"], prompt=prompt(by[x["id"]]), reply=x.get("reply", by[x["id"]]["base"])))
        elif isinstance(x, dict) and "user" in x:
            out.append(dict(key=x["user"][:60], prompt=prompt(x["user"]), reply=x.get("reply", "")))
        else:
            raise ValueError(f"unsupported input {x!r}")
    return out


def _passes_for(n):
    return max(1, math.ceil(n / MAX_SEQS))


# ====================================================================== free tools: weights
def weight_names(pattern=None):
    """Names of every editable tensor (optionally filtered by a regex) with shapes. Free."""
    out = {n: list(p.shape) for n, p in _params().items() if _in_scope(n) and (not pattern or re.search(pattern, n))}
    _log("weight_names", pattern=pattern, n=len(out))
    return out


def _sel(t, rows, cols):
    if rows is not None:
        t = t[torch.as_tensor(rows) if not isinstance(rows, slice) else rows]
    if cols is not None:
        t = t[:, torch.as_tensor(cols) if not isinstance(cols, slice) else cols]
    return t


def weight(name, rows=None, cols=None, edited=False):
    """A float32 CPU copy of weight `name` (or rows / cols of it: int lists, ranges or slices). edited=True returns it
    with the current edit applied. Free. (Gemma RMSNorm weights are used as (1 + w).)"""
    p = _params()
    if name not in p:
        raise KeyError(f"unknown tensor {name}; see weight_names()")
    t = p[name].detach()
    if edited:
        e = _load_sparse().get(name)
        if e is not None:
            t = t.clone()
            t.view(-1)[e[0]] = e[1]
    t = _sel(t, rows, cols).float().clone()
    _log("weight", name=name, rows=_short(rows), cols=_short(cols), edited=edited, shape=list(t.shape))
    return t


def _short(x):
    if x is None or isinstance(x, (int, slice)):
        return str(x) if x is not None else None
    x = list(x)
    return x[:10] + (["..."] if len(x) > 10 else [])


# ====================================================================== the edit (sparse)
def _load_sparse():
    mt = EDIT.stat().st_mtime_ns if EDIT.exists() else None
    if _S["edit"] is None or _S.get("edit_mtime") != mt:   # another process may have changed the edit
        e = {}
        if EDIT.exists():
            from safetensors.torch import load_file
            raw = load_file(str(EDIT))
            for k in raw:
                if k.endswith("::idx"):
                    n = k[:-5]
                    e[n] = (raw[k].long(), raw[n + "::val"])
        _S["edit"], _S["edit_mtime"] = e, mt
    return _S["edit"]


def _write_sparse(e, path=EDIT):
    from safetensors.torch import save_file
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    flat = {}
    for n, (i, v) in e.items():
        flat[n + "::idx"] = i.contiguous()
        flat[n + "::val"] = v.contiguous()
    save_file(flat, str(path), metadata=dict(format="patchkit-sparse-v1"))
    if Path(path) == EDIT:
        _S["edit"], _S["edit_mtime"] = e, EDIT.stat().st_mtime_ns


def _count(e):
    return sum(int(i.numel()) for i, _ in e.values())


def _update(name, fn, note, how):
    """Apply fn(full_edited_tensor_float32) -> new tensor for one parameter, re-diff against the original, validate
    the budget, save. Returns entries changed in total."""
    p = _params()
    if name not in p:
        raise KeyError(f"unknown tensor {name}; see weight_names()")
    if not _in_scope(name):
        raise ValueError(f"{name} is out of scope (only the language model may be edited)")
    base = p[name].detach()
    e = dict(_load_sparse())
    cur = base.clone()
    if name in e:
        cur.view(-1)[e[name][0]] = e[name][1]
    new = fn(cur.float())
    if tuple(new.shape) != tuple(base.shape):
        raise ValueError(f"shape mismatch for {name}: {tuple(new.shape)} vs {tuple(base.shape)}")
    if not torch.isfinite(new).all():
        raise ValueError("new values contain NaN/inf")
    new = new.to(base.dtype).reshape(-1)
    diff = (new != base.reshape(-1)).nonzero().squeeze(1)
    if diff.numel():
        e[name] = (diff, new[diff].clone())
    else:
        e.pop(name, None)
    total = _count(e)
    if total > _cfg()["max_changed"]:
        _log("edit_rejected", how=how, name=name, would_change=total, note=note)
        raise ValueError(f"edit would change {total:,} entries; the limit is {_cfg()['max_changed']:,}. Not saved.")
    _write_sparse(e)
    _S["edit"] = e
    dn = float((new.float() - base.reshape(-1).float()).norm())
    _log("write_edit", how=how, name=name, tensor_changed=int(diff.numel()), total_changed=total,
         tensor_l2_change=round(dn, 5), note=note)
    print(f"[edit] {name}: {int(diff.numel()):,} entries differ (L2 change {dn:.4g}); total {total:,} / "
          f"{_cfg()['max_changed']:,}", flush=True)
    return total


def _vals(values, shape):
    v = torch.as_tensor(values, dtype=torch.float32) if not torch.is_tensor(values) else values.float().cpu()
    return v.expand(shape) if v.shape != shape else v


def set_rows(name, rows, values, note=""):
    """Set whole rows of tensor `name` (rows: int or list of ints; values: [len(rows), n_cols] or broadcastable).
    For a Linear weight [out, in], row r is everything output unit r reads. Free. Returns total entries changed."""
    rows = [rows] if isinstance(rows, int) else list(rows)

    def fn(t):
        t[rows] = _vals(values, t[rows].shape)
        return t
    return _update(name, fn, note, f"rows {_short(rows)}")


def set_cols(name, cols, values, note=""):
    """Set whole columns of a 2-D tensor (values: [n_rows, len(cols)] or broadcastable; a 1-D vector of length
    n_rows for a single column). For a Linear weight [out, in], column c is what input unit c writes, e.g. a
    down_proj column is one MLP neuron's output vector. Free."""
    cols = [cols] if isinstance(cols, int) else list(cols)

    def fn(t):
        v = values if torch.is_tensor(values) else torch.as_tensor(values, dtype=torch.float32)
        if v.dim() == 1 and len(cols) == 1 and v.numel() == t.shape[0]:
            v = v[:, None]
        t[:, cols] = _vals(v, t[:, cols].shape)
        return t
    return _update(name, fn, note, f"cols {_short(cols)}")


def set_entries(name, index, values, note=""):
    """Set individual entries. index: a tuple of index lists (one per dim, like t[index]) or a 1-D list of flat
    indices; values: matching shape or a scalar. Free."""

    def fn(t):
        if isinstance(index, tuple):
            ix = tuple(torch.as_tensor(i) for i in index)
            t[ix] = _vals(values, t[ix].shape)
        else:
            ix = torch.as_tensor(index).long()
            f = t.reshape(-1)
            f[ix] = _vals(values, f[ix].shape)
            t = f.view_as(t)
        return t
    return _update(name, fn, note, "entries")


def set_tensor(name, new, note=""):
    """Replace the whole tensor `name` with `new` (same shape); only entries that differ count. Free."""
    return _update(name, lambda t: torch.as_tensor(new).float().cpu(), note, "tensor")


def clear_edit(name=None):
    """Drop the whole current edit, or just its changes to tensor `name`. Free."""
    e = dict(_load_sparse())
    if name is None:
        e = {}
    else:
        e.pop(name, None)
    _write_sparse(e)
    _S["edit"] = e
    _log("clear_edit", name=name, total_changed=_count(e))
    print(f"[edit] cleared {'all' if name is None else name}; total {_count(e):,} entries")


def current_edit():
    """The current edit as {name: full new tensor (bf16, CPU)}. Free."""
    p, out = _params(), {}
    for n, (i, v) in _load_sparse().items():
        t = p[n].detach().clone()
        t.view(-1)[i] = v
        out[n] = t
    return out


def edit_summary(quiet=False):
    """Per tensor: entries changed and L2 norm of the change; plus the total. Free."""
    p, per = _params(), {}
    for n, (i, v) in _load_sparse().items():
        d = v.float() - p[n].detach().reshape(-1)[i].float()
        per[n] = dict(entries=int(i.numel()), l2_change=round(float(d.norm()), 5),
                      l2_original=round(float(p[n].detach().reshape(-1)[i].float().norm()), 5))
    out = dict(entries_changed=sum(x["entries"] for x in per.values()), max_changed=_cfg()["max_changed"],
               tensors=per)
    if not quiet:
        print(json.dumps(out, indent=1))
        _log("edit_summary", entries_changed=out["entries_changed"])
    return out


def save_copy(tag):
    """Keep a copy of the current edit as work/edits/<tag>.safetensors. Free."""
    _write_sparse(_load_sparse(), COPIES / f"{tag}.safetensors")
    _log("save_copy", tag=tag, entries=_count(_load_sparse()))
    print(f"[edit] copy saved: work/edits/{tag}.safetensors")


def load_copy(tag):
    """Make a saved copy the current edit again. Free."""
    from safetensors.torch import load_file
    raw = load_file(str(COPIES / f"{tag}.safetensors"))
    e = {k[:-5]: (raw[k].long(), raw[k[:-5] + "::val"]) for k in raw if k.endswith("::idx")}
    if _count(e) > _cfg()["max_changed"]:
        raise ValueError("copy exceeds the edit budget")
    _write_sparse(e)
    _S["edit"] = e
    _log("load_copy", tag=tag, entries=_count(e))
    print(f"[edit] current edit = copy {tag} ({_count(e):,} entries)")


def list_copies():
    """Tags of saved edit copies. Free."""
    return sorted(p.stem for p in COPIES.glob("*.safetensors"))


# ====================================================================== GPU context
@contextlib.contextmanager
def _gpu(kind, n_passes, edited):
    """Take the repo GPU lock, charge passes, move the model to the GPU with (optionally) the current edit applied,
    and undo everything afterwards."""
    from gpu_lock import gpu
    m = _model()
    _check(n_passes, kind)
    t_wait = time.time()
    with gpu(f"agent {SESSION.name} {kind}"):
        _S["wait_s"] = round(time.time() - t_wait, 1)
        _spend(n_passes, kind)
        orig = {}
        try:
            m.to(L.DEV)
            if edited:
                p = dict(m.named_parameters())
                with torch.no_grad():
                    for n, (i, v) in _load_sparse().items():
                        flat = p[n].data.view(-1)
                        ii = i.to(flat.device)
                        orig[n] = (ii, flat[ii].clone())
                        flat[ii] = v.to(flat.device, flat.dtype)
            yield m
        finally:
            p = dict(m.named_parameters())
            with torch.no_grad():
                for n, (ii, ov) in orig.items():
                    p[n].data.view(-1)[ii] = ov
            m.to("cpu")
            torch.cuda.empty_cache()


def _tm(model):
    return L.text_model(model)


# ====================================================================== counted tools
def generate(inputs, max_new=None, edited=True, prefix=None):
    """Greedy replies, decoded exactly as scored. 1 pass per 64 prompts.
    inputs (a list; or one item):
      int                         a practice id (its scored prompt)
      str                         a user message of your own (rendered with the task's chat template / system prompt)
      {"user": str}               same as str
      {"raw": str}                an exact, already chat-formatted prompt string
    max_new: default the task's scoring length (at most 512). prefix: force the reply to start with this text.
    edited: True = with the current edit (identical to the original model when no edit is saved), False = original.
    Returns [{input, reply}]."""
    tok, t = tokenizer(), _task()
    xs = _inputs(inputs)
    max_new = min(int(max_new or t.max_new), MAX_NEW_CAP)
    n = _passes_for(len(xs))
    t0 = time.time()
    with _gpu("generate", n, edited) as m:
        outs = L.generate(m, tok, [x["prompt"] for x in xs], max_new=max_new, batch=min(MAX_SEQS, t.gen_batch),
                          prefix=prefix)
    res = [dict(input=x["key"], reply=o) for x, o in zip(xs, outs)]
    _log("generate", n_inputs=len(xs), inputs=[x["key"] for x in xs][:64], max_new=max_new, edited=edited,
         prefix=prefix, passes=n, seconds=round(time.time() - t0, 1), gpu_wait_s=_S.get("wait_s"), edit_entries=_count(_load_sparse()))
    return res


def _positions(tok, xs, positions):
    """-> seqs, per-seq position lists, per-seq prompt lengths."""
    seqs, pos, plens = [], [], []
    for k, x in enumerate(xs):
        full = x["prompt"] + x["reply"]
        n_p = len(tok(x["prompt"], add_special_tokens=False).input_ids)
        n = len(tok(full, add_special_tokens=False).input_ids)
        if isinstance(positions, (list, tuple)) and not isinstance(positions, str):
            ps = positions[k]
            ps = [ps] if isinstance(ps, int) else list(ps)
            ps = [p if p >= 0 else n + p for p in ps]
        elif positions == "last_prompt":
            ps = [n_p - 1]
        elif positions == "reply":
            ps = list(range(n_p, n))
        elif positions == "all":
            ps = list(range(n))
        else:
            raise ValueError("positions: 'last_prompt', 'reply', 'all', or a list (one entry per input) of index lists")
        if any(p < 0 or p >= n for p in ps):
            raise ValueError(f"position out of range for input {x['key']} (length {n})")
        seqs.append(full)
        pos.append(ps)
        plens.append(n_p)
    return seqs, pos, plens


def _hooks(model, layers, kinds, cur):
    tm, hs = _tm(model), []
    for Lr in layers:
        lay = tm.layers[Lr]
        if "resid" in kinds:
            hs.append(lay.register_forward_hook(
                lambda mod, i, o, Lr=Lr: cur.__setitem__(f"resid@{Lr}", o[0] if isinstance(o, tuple) else o)))
        if "attn_out" in kinds:
            hs.append(lay.self_attn.register_forward_hook(
                lambda mod, i, o, Lr=Lr: cur.__setitem__(f"attn_out@{Lr}", o[0] if isinstance(o, tuple) else o)))
        if "mlp_in" in kinds:
            hs.append(L.mlp_in_norm(lay).register_forward_hook(
                lambda mod, i, o, Lr=Lr: cur.__setitem__(f"mlp_in@{Lr}", o)))
        if "mlp_act" in kinds:
            hs.append(lay.mlp.down_proj.register_forward_hook(
                lambda mod, i, o, Lr=Lr: cur.__setitem__(f"mlp_act@{Lr}", i[0])))
        if "mlp_out" in kinds:
            hs.append(lay.mlp.register_forward_hook(lambda mod, i, o, Lr=Lr: cur.__setitem__(f"mlp_out@{Lr}", o)))
    return hs


KINDS = ("resid", "attn_out", "mlp_in", "mlp_act", "mlp_out")


@torch.no_grad()
def activations(inputs, layers, positions="last_prompt", kinds=("resid",), edited=True, logits=False):
    """Teacher-forced forward pass; returns activations at chosen token positions. 1 pass per 64 sequences.
    inputs: as in generate(), plus {"id": int, "reply": str} or {"user": str, "reply": str} to teacher-force a reply
      of your choice. A practice id is teacher-forced with the original model's own practice reply.
    layers: decoder layer indices (negative counts from the end).
    positions: "last_prompt" (the token right before the reply starts: where the first reply token is decided),
      "reply" (every reply token), "all", or a list with one int / index list per input (negative = from the end).
    kinds (per layer):
      resid     residual stream after the layer                                   [d_model]
      attn_out  self-attention output (before the post-attention norm)            [d_model]
      mlp_in    the normalized MLP input (output of the pre-MLP norm)             [d_model]
      mlp_act   MLP hidden activation act(gate) * up, the input of down_proj      [d_mlp]
      mlp_out   MLP output (down_proj; Gemma then applies post_feedforward_layernorm) [d_model]
    logits=True also returns the output log-probs at those positions ("logprobs", [N, vocab] float16; costly).
    Returns dict: "<kind>@<layer>" -> float32 tensor [N, dim] (rows = all selected positions of all inputs, in order),
      "rows" -> [(input, position, token at position, next token)], "prompt_len" -> per input."""
    tok = tokenizer()
    xs = _inputs(inputs)
    nl = len(_tm(_model()).layers)
    layers = [int(Lr) % nl for Lr in ([layers] if isinstance(layers, int) else layers)]
    kinds = [kinds] if isinstance(kinds, str) else list(kinds)
    bad = [k for k in kinds if k not in KINDS]
    if bad:
        raise ValueError(f"unknown kinds {bad}; choose from {KINDS}")
    seqs, pos, plens = _positions(tok, xs, positions)
    cfg = _model().config.get_text_config() if hasattr(_model().config, "get_text_config") else _model().config
    dims = dict(resid=cfg.hidden_size, attn_out=cfg.hidden_size, mlp_in=cfg.hidden_size, mlp_out=cfg.hidden_size,
                mlp_act=cfg.intermediate_size)
    n_rows = sum(len(p) for p in pos)
    floats = n_rows * (sum(dims[k] for k in kinds) * len(layers) + (cfg.vocab_size // 2 if logits else 0))
    if floats > MAX_ACT_FLOATS:
        raise ValueError(f"{floats:,} activation entries > {MAX_ACT_FLOATS:,}: ask for fewer positions/layers/kinds")
    n = _passes_for(len(xs))
    res = {f"{k}@{Lr}": [] for Lr in layers for k in kinds}
    lps, rows = [], []
    t0 = time.time()
    with _gpu("activations", n, edited) as m:
        cur = {}
        hs = _hooks(m, layers, kinds, cur)
        tok.padding_side = "right"
        try:
            for i in range(0, len(seqs), FWD_BATCH):
                enc = tok(seqs[i:i + FWD_BATCH], return_tensors="pt", padding=True, add_special_tokens=False).to(L.DEV)
                out = m(**enc, use_cache=False)
                for b in range(enc.input_ids.shape[0]):
                    ps = pos[i + b]
                    for k, v in cur.items():
                        res[k].append(v[b, ps].float().cpu())
                    if logits:
                        lps.append(torch.log_softmax(out.logits[b, ps].float(), -1).half().cpu())
                    ids = enc.input_ids[b].tolist()
                    nb = int(enc.attention_mask[b].sum())
                    for p in ps:
                        rows.append((xs[i + b]["key"], p, tok.decode([ids[p]]),
                                     tok.decode([ids[p + 1]]) if p + 1 < nb else None))
                cur.clear()
        finally:
            tok.padding_side = "left"
            for h in hs:
                h.remove()
    out = {k: torch.cat(v) for k, v in res.items()}
    if logits:
        out["logprobs"] = torch.cat(lps)
    out["rows"], out["prompt_len"] = rows, plens
    _log("activations", n_inputs=len(xs), inputs=[x["key"] for x in xs][:64], layers=layers, kinds=kinds,
         positions=positions if isinstance(positions, str) else "custom", rows=n_rows, edited=edited, passes=n,
         seconds=round(time.time() - t0, 1), gpu_wait_s=_S.get("wait_s"))
    return out


def _tid(tok, s):
    if s in ("<eot>", "<end_of_turn>"):
        return L.eot_id(tok)
    ids = tok.encode(s, add_special_tokens=False)
    return ids[0]


@torch.no_grad()
def logit_lens(inputs, layers=None, position="last_prompt", k=10, watch=(), edited=True):
    """What the model would predict next at one position per input, read from the residual stream after each layer
    (final norm + output matrix applied to it: the "logit lens"), plus the real output ("final").
    1 pass per 64 inputs. position: "last_prompt" (the first reply token's decision), or one int per input in a list.
    watch: strings (first token of each is tracked; "<eot>" = end of turn) whose log-prob is reported at every layer.
    Returns [{input, position, token, layers: {L or "final": {"top": [(token, prob)], "watch": {s: logprob}}}}]."""
    tok = tokenizer()
    xs = _inputs(inputs)
    nl = len(_tm(_model()).layers)
    layers = list(range(0, nl, max(1, nl // 8))) if layers is None else [int(Lr) % nl for Lr in layers]
    pos_arg = "last_prompt" if position == "last_prompt" else [[p] for p in position]
    seqs, pos, _ = _positions(tok, xs, pos_arg)
    wid = {s: _tid(tok, s) for s in watch}
    n = _passes_for(len(xs))
    res = []
    t0 = time.time()
    with _gpu("logit_lens", n, edited) as m:
        cur = {}
        hs = _hooks(m, layers, ["resid"], cur)
        tm, W = _tm(m), m.get_output_embeddings().weight
        tok.padding_side = "right"
        try:
            for i in range(0, len(seqs), FWD_BATCH):
                enc = tok(seqs[i:i + FWD_BATCH], return_tensors="pt", padding=True, add_special_tokens=False).to(L.DEV)
                out = m(**enc, use_cache=False)
                for b in range(enc.input_ids.shape[0]):
                    p = pos[i + b][0]
                    rec = dict(input=xs[i + b]["key"], position=p, token=tok.decode([enc.input_ids[b, p].item()]),
                               layers={})
                    vecs = [(Lr, (tm.norm(cur[f"resid@{Lr}"][b, p]) @ W.T).float()) for Lr in layers]
                    vecs.append(("final", out.logits[b, p].float()))
                    for key, lg in vecs:
                        lp = torch.log_softmax(lg, -1)
                        top = lp.topk(k)
                        rec["layers"][key] = dict(
                            top=[(tok.decode([j]), round(math.exp(v), 4)) for v, j in
                                 zip(top.values.tolist(), top.indices.tolist())],
                            watch={s: round(lp[j].item(), 3) for s, j in wid.items()})
                    res.append(rec)
                cur.clear()
        finally:
            tok.padding_side = "left"
            for h in hs:
                h.remove()
    _log("logit_lens", n_inputs=len(xs), inputs=[x["key"] for x in xs][:64], layers=layers, watch=list(watch),
         edited=edited, passes=n, seconds=round(time.time() - t0, 1), gpu_wait_s=_S.get("wait_s"))
    return res


@torch.no_grad()
def unit_usage(inputs, layer, edited=True, positions="all"):
    """Per-neuron statistics of the MLP hidden activation (act(gate) * up) of one layer over all selected tokens of
    the inputs (default: every token of prompt + teacher-forced reply). 1 pass per 64 sequences.
    Returns dict(mean_abs, mean, max_abs: tensors [d_mlp]; n_tokens)."""
    A = None
    acts = activations(inputs, [layer], positions=positions, kinds=("mlp_act",), edited=edited)
    A = next(v for k, v in acts.items() if k.startswith("mlp_act@"))
    out = dict(mean_abs=A.abs().mean(0), mean=A.mean(0), max_abs=A.abs().max(0).values, n_tokens=len(A))
    _log("unit_usage", layer=layer, n_tokens=len(A))
    return out


def test(ids=None, n=128, seed=0, edited=True, max_new=None):
    """The task's own metrics (task.score, headline included) on practice items, generated greedily as scored.
    ids: practice ids; default a fixed stratified sample of n practice items (n="all" for every one).
    1 pass per 64 items. The original model's scores on the same items come free from its stored practice replies.
    Returns dict(n, ids, headline, edited, base, replies=[{id, label, base, edited}]); also writes
    work/test_<k>.jsonl with every reply."""
    t = _task()
    if ids is None:
        items = TK.subset(practice(), None if n in (None, "all") else int(n), seed, lambda x: x["label"])
    else:
        by = _by_id()
        items = [by[i] for i in ids]
    outs = generate([x["id"] for x in items], max_new=max_new, edited=edited)
    texts = [o["reply"] for o in outs]
    sc_e = t.score(items, texts)
    sc_b = t.score(items, [x["base"] for x in items])
    h = list(t.headline)
    rows = [dict(id=x["id"], label=x["label"], base=x["base"], edited=s) for x, s in zip(items, texts)]
    k = len(list(WORK.glob("test_*.jsonl")))
    L.write_jsonl(WORK / f"test_{k}.jsonl", rows)
    res = dict(n=len(items), ids=[x["id"] for x in items], headline={x: dict(base=sc_b.get(x), edited=sc_e.get(x))
                                                                      for x in h},
               edited=sc_e, base=sc_b, replies=rows, file=f"work/test_{k}.jsonl")
    _log("test", n=len(items), edited=edited, headline=res["headline"], file=res["file"],
         edit_entries=_count(_load_sparse()))
    print(f"[test] n={len(items)} " + ", ".join(f"{x}: base {sc_b.get(x)} -> {'edited' if edited else 'base'} "
                                                f"{sc_e.get(x)}" for x in h), flush=True)
    return res


# ====================================================================== info
def info():
    """Model summary, tensor names/shapes (layer 0 shown as a pattern), budgets, practice summary. Free."""
    m, c, t = _model(), _cfg(), _task()
    cfg = m.config.get_text_config() if hasattr(m.config, "get_text_config") else m.config
    names = {}
    for k, p in m.named_parameters():
        if not _in_scope(k):
            continue
        if re.search(r"\.layers\.\d+\.", k) and not re.search(r"\.layers\.0\.", k):
            continue
        names[re.sub(r"\.layers\.0\.", ".layers.{i}.", k)] = list(p.shape)
    tok = tokenizer()
    out_w = m.get_output_embeddings().weight
    tied = any(out_w is p for n, p in m.named_parameters() if n.endswith("embed_tokens.weight"))
    P = practice()
    s = dict(model=c["model"], task=c["task"], architecture=type(_tm(m)).__name__, layers=cfg.num_hidden_layers,
             d_model=cfg.hidden_size, d_mlp=cfg.intermediate_size, heads=cfg.num_attention_heads,
             kv_heads=cfg.num_key_value_heads, vocab=cfg.vocab_size, tied_embeddings=tied,
             params_in_scope=c["n_params"], tensors=names,
             special_tokens=dict(eot=tok.decode([L.eot_id(tok)]), eot_id=L.eot_id(tok), bos=tok.bos_token),
             budgets=dict(passes=c["passes"], max_changed=c["max_changed"], minutes=c["minutes"],
                          prompts_per_pass=MAX_SEQS),
             scoring=dict(max_new=t.max_new, decoding="greedy", headline=list(t.headline)),
             practice=dict(n=len(P), label_1=sum(x["label"] == 1 for x in P), label_0=sum(x["label"] == 0 for x in P),
                           unlabelled=sum(x["label"] is None for x in P)))
    print(json.dumps(s, indent=1, ensure_ascii=False))
    _log("info")
    return s


# ====================================================================== CLI
def _parse_inputs(toks):
    """CLI inputs: ints = practice ids; pos:N / neg:N / any:N = the first N practice ids with label 1 / 0 / any;
    @file.json = a JSON list of inputs; anything else = a user message."""
    out = []
    for s in toks:
        if re.fullmatch(r"-?\d+", s):
            out.append(int(s))
        elif re.fullmatch(r"(pos|neg|any):\d+", s):
            lab, k = s.split(":")
            out += [x["id"] for x in practice({"pos": 1, "neg": 0, "any": "any"}[lab])][:int(k)]
        elif s.startswith("@"):
            out += json.loads(Path(s[1:]).read_text(encoding="utf-8"))
        else:
            out.append(s)
    return out


def _ints(s):
    if s is None:
        return None
    if ":" in s:
        a, b = s.split(":")
        return list(range(int(a), int(b)))
    return [int(x) for x in s.split(",")]


def main(argv=None):
    ap = argparse.ArgumentParser(prog="tools.py", description="Agent session tools (see PROMPT.md).")
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("info")
    sp.add_parser("budget")
    q = sp.add_parser("practice"); q.add_argument("--label", default="any"); q.add_argument("--n", type=int, default=5)
    q = sp.add_parser("prompt"); q.add_argument("x")
    q = sp.add_parser("names"); q.add_argument("--filter")
    q = sp.add_parser("weight"); q.add_argument("name"); q.add_argument("--rows"); q.add_argument("--cols")
    q.add_argument("--edited", action="store_true"); q.add_argument("--out")
    for c in ("generate", "activations", "lens", "usage"):
        q = sp.add_parser(c)
        q.add_argument("inputs", nargs="+")
        q.add_argument("--base", action="store_true", help="run the original model instead of the current edit")
        if c == "generate":
            q.add_argument("--max-new", type=int); q.add_argument("--prefix")
        if c in ("activations", "lens"):
            q.add_argument("--layers", required=c == "activations")
        if c == "activations":
            q.add_argument("--positions", default="last_prompt"); q.add_argument("--kinds", default="resid")
        if c == "lens":
            q.add_argument("--k", type=int, default=10); q.add_argument("--watch", default="")
        if c == "usage":
            q.add_argument("--layer", type=int, required=True)
        if c in ("activations", "usage"):
            q.add_argument("--out", required=True, help="a .pt file under work/")
    q = sp.add_parser("test"); q.add_argument("--n", default="128"); q.add_argument("--ids"); q.add_argument("--base", action="store_true")
    q = sp.add_parser("write", help="set rows / cols / entries of a tensor from a .pt file or a number")
    q.add_argument("name"); q.add_argument("--rows"); q.add_argument("--cols"); q.add_argument("--flat")
    q.add_argument("--values", required=True, help="a number or a .pt file holding a tensor"); q.add_argument("--note", default="")
    sp.add_parser("edit-summary")
    q = sp.add_parser("clear-edit"); q.add_argument("--name")
    q = sp.add_parser("save-copy"); q.add_argument("tag")
    q = sp.add_parser("load-copy"); q.add_argument("tag")
    a = ap.parse_args(argv)
    pj = lambda x: print(json.dumps(x, indent=1, ensure_ascii=False, default=str))  # noqa: E731
    if a.cmd == "info":
        info()
    elif a.cmd == "budget":
        budget()
    elif a.cmd == "practice":
        lab = {"any": "any", "1": 1, "0": 0, "None": None}[a.label]
        xs = practice(lab)
        print(f"{len(xs)} items")
        pj([dict(x, base=x["base"][:300]) for x in xs[:a.n]])
        _log("practice", label=a.label)
    elif a.cmd == "prompt":
        print(prompt(int(a.x) if re.fullmatch(r"\d+", a.x) else a.x))
    elif a.cmd == "names":
        pj(weight_names(a.filter))
    elif a.cmd == "weight":
        w = weight(a.name, _ints(a.rows), _ints(a.cols), edited=a.edited)
        print(f"shape {list(w.shape)} mean {w.mean():.4g} std {w.std() if w.numel() > 1 else 0:.4g} "
              f"min {w.min():.4g} max {w.max():.4g} L2 {w.norm():.4g}")
        if a.out:
            torch.save(w, a.out); print("saved", a.out)
        elif w.numel() <= 64:
            print(w.tolist())
    elif a.cmd == "generate":
        pj(generate(_parse_inputs(a.inputs), max_new=a.max_new, edited=not a.base, prefix=a.prefix))
    elif a.cmd == "activations":
        r = activations(_parse_inputs(a.inputs), _ints(a.layers), positions=a.positions, kinds=a.kinds.split(","),
                        edited=not a.base)
        torch.save(r, a.out)
        print("saved", a.out, {k: list(v.shape) for k, v in r.items() if torch.is_tensor(v)}, f"rows={len(r['rows'])}")
    elif a.cmd == "lens":
        pj(logit_lens(_parse_inputs(a.inputs), _ints(a.layers) if a.layers else None, k=a.k,
                      watch=[w for w in a.watch.split(",") if w], edited=not a.base))
    elif a.cmd == "usage":
        r = unit_usage(_parse_inputs(a.inputs), a.layer, edited=not a.base)
        torch.save(r, a.out)
        ma = r["mean_abs"]
        print(f"saved {a.out}; n_tokens {r['n_tokens']}; least-used units {ma.argsort()[:10].tolist()}; "
              f"most-used {ma.argsort(descending=True)[:10].tolist()}")
    elif a.cmd == "test":
        r = test(ids=_ints(a.ids), n=a.n, edited=not a.base)
        pj(dict(n=r["n"], headline=r["headline"], edited=r["edited"], base=r["base"], file=r["file"]))
    elif a.cmd == "write":
        v = torch.load(a.values) if a.values.endswith(".pt") else float(a.values)
        if a.rows:
            set_rows(a.name, _ints(a.rows), v, note=a.note)
        elif a.cols:
            set_cols(a.name, _ints(a.cols), v, note=a.note)
        elif a.flat:
            set_entries(a.name, _ints(a.flat), v, note=a.note)
        else:
            set_tensor(a.name, v, note=a.note)
    elif a.cmd == "edit-summary":
        edit_summary()
    elif a.cmd == "clear-edit":
        clear_edit(a.name)
    elif a.cmd == "save-copy":
        save_copy(a.tag)
    elif a.cmd == "load-copy":
        load_copy(a.tag)


if __name__ == "__main__":
    main()
