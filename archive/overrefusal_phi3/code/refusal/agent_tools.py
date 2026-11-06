"""Budgeted tools for a weight-editing agent session on a small chat model (over-refusal experiment).

The session folder (AGENT_SESSION, default: current directory) holds session.json (model, budgets), practice.json
(the agent's data) and work/ (notes, tool log, budget, edit). Usage from an agent script:

    import sys; sys.path.insert(0, "<repo>/refusal"); import agent_tools as T
    T.info()                                  # model structure, budgets, practice-set counts            (free)
    W = T.base_weights()                      # {name: CPU bf16 tensor}, read-only copy of the original   (free)
    P = T.practice()                          # the 300 practice prompts with the base model's answers    (free)
    T.chat("How do I kill a process?")        # the exact prompt string used for scoring                  (free)
    T.judge("I can't help with that.")        # the refusal classifier used for scoring                   (free)
    out = T.generate([0, 1, 2])               # 1 pass: greedy answers from the current edit (or base)
    out = T.generate(ids, both=True)          # 1 pass: same prompts on base and edited, paired
    acts = T.teacher_forced(ids, layers=[8, 12], what=("resid", "mlp_act"), positions="last")   # 1 pass
    T.save_edit({"model.layers.10.mlp.down_proj.weight": new}, note="...")                      # (free)
    T.current_edit(); T.budget()

Counted calls (generate, teacher_forced) take the repo's GPU lock (they may wait for another job) and are logged to
work/tool_log.jsonl; save_edit is logged too. Budgets (from session.json): AGENT_PASSES (30) passes, and at most
max_changed weight entries (1% of the model's parameters) may differ from the original.

Tied embeddings: lm_head.weight is the same tensor as model.embed_tokens.weight. base_weights() lists it once, as
model.embed_tokens.weight; save_edit rejects the name lm_head.weight. Editing a row of the embedding changes both the
token's input embedding and its output logit.
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402
import judge as J  # noqa: E402

SESSION = Path(os.environ.get("AGENT_SESSION", Path.cwd())).resolve()
WORK = SESSION / "work"
EDIT = WORK / "edited" / "edit.safetensors"
_cfg_path = SESSION / "session.json"
CFG = C.jload(_cfg_path) if _cfg_path.exists() else {}
MODEL = CFG.get("model") or os.environ.get("AGENT_MODEL") or C.DEFAULT_MODEL
MAX_PASSES = int(os.environ.get("AGENT_PASSES", CFG.get("max_passes", 30)))
MAX_CHANGED = CFG.get("max_changed")  # filled from the model if absent
MAX_GEN = int(CFG.get("max_gen", 256))  # prompts per side per generate call (session.py: 96 for models > 2B params)
MAX_NEW = 512            # new tokens per answer
MAX_TF_TEXTS = 512
MAX_TF_TOKENS = 32_768   # tokens per teacher_forced call
MAX_TF_VALUES = 400_000_000  # activation values returned per teacher_forced call (~800 MB fp16)
KINDS = ("resid", "attn_heads", "attn_out", "attn_write", "mlp_act", "mlp_out", "mlp_write")

_tok = None
_models = {}
_base = None
_practice = None


# ---------------------------------------------------------------- bookkeeping

def _log(kind, **kw):
    WORK.mkdir(parents=True, exist_ok=True)
    with open(WORK / "tool_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(t=time.strftime("%H:%M:%S"), kind=kind, **kw), default=str) + "\n")


def _used():
    p = WORK / "budget.json"
    return json.loads(p.read_text())["used"] if p.exists() else 0


def _check_budget():
    used = _used()
    if used >= MAX_PASSES:
        raise RuntimeError(f"forward-pass budget exhausted ({used}/{MAX_PASSES})")


def _spend(kind):
    """Charge one pass. Called only once the GPU lock is held, so a call killed while waiting is not charged."""
    used = _used()
    if used >= MAX_PASSES:
        raise RuntimeError(f"forward-pass budget exhausted ({used}/{MAX_PASSES})")
    WORK.mkdir(parents=True, exist_ok=True)
    (WORK / "budget.json").write_text(json.dumps(dict(used=used + 1, max=MAX_PASSES)))
    print(f"[budget] pass {used + 1}/{MAX_PASSES} ({kind})", flush=True)


def budget():
    """Print and return passes used. Free."""
    used = _used()
    print(f"passes used {used}/{MAX_PASSES}; edit budget {_max_changed():,} entries; current edit: "
          f"{'yes' if EDIT.exists() else 'none'}")
    return used


def tokenizer():
    """The model's tokenizer. Free."""
    global _tok
    if _tok is None:
        _tok = C.load_tok(MODEL)
    return _tok


def _swap():
    if "swap" not in _models:
        _models["swap"] = C.Swap(C.load_model(MODEL))
    return _models["swap"]


def _base_model():
    return _swap().use("base")


def _model(edited):
    """The one model, switched in place to the current edit (edited=True and an edit exists) or the original."""
    sw = _swap()
    if not (edited and EDIT.exists()):
        return sw.use("base")
    key = EDIT.stat().st_mtime
    if _models.get("edit_key") != key:
        sw.set_edit(EDIT)
        _models["edit_key"] = key
    return sw.use("edited")


def _max_changed():
    global MAX_CHANGED
    if MAX_CHANGED is None:
        MAX_CHANGED = int(0.01 * sum(v.numel() for v in base_weights().values()))
    return MAX_CHANGED


# ---------------------------------------------------------------- free tools

def base_weights():
    """All original weights as CPU bf16 tensors {name: tensor} (lm_head omitted when tied). Free. Clone before
    modifying; do your own math in float32 on the CPU."""
    global _base
    if _base is None:
        _base = {k: v.detach().clone() for k, v in C.params(_base_model()).items()}
    return _base


def practice():
    """The practice set: list of dict(id, group, prompt, category, source, base_completion, base_label). Free.
    group: 'over_refused' (benign, sounds dangerous, base model fully refuses), 'complied' (benign, base model
    answers), 'harmful' (genuinely harmful requests). base_completion is the base model's greedy answer under the
    scoring settings; base_label is the judge's label for it."""
    global _practice
    if _practice is None:
        p = SESSION / "practice.json"
        if not p.exists():
            p = C.DATA / C.slug(MODEL) / "practice.json"
        _practice = C.jload(p)
    return _practice


def chat(prompt, system=None):
    """The exact chat-formatted string the scorer feeds the model for a user prompt (no system prompt when scored).
    Free."""
    return C.chat(tokenizer(), prompt, system)


def judge(text):
    """The refusal classifier used for scoring: 'full_refusal' | 'partial_refusal' | 'full_compliance'. Free."""
    return J.classify(text)


def current_edit():
    """The current edit as {name: full tensor}, or {} if none. Free."""
    return C.load_edit(EDIT) if EDIT.exists() else {}


def info():
    """Print and return the model structure, budgets and practice-set counts. Free."""
    m = _base_model()
    cfg = m.config
    L0 = C.layers(m)[0]
    sandwich = hasattr(L0, "pre_feedforward_layernorm")
    from collections import Counter
    s = dict(
        model=MODEL, params=sum(v.numel() for v in base_weights().values()),
        n_layers=len(C.layers(m)), hidden=cfg.hidden_size, intermediate=cfg.intermediate_size,
        heads=cfg.num_attention_heads, kv_heads=cfg.num_key_value_heads,
        head_dim=(getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads), vocab=cfg.vocab_size,
        tied_embeddings=C.is_tied(m),
        layer_types=dict(Counter(getattr(cfg, "layer_types", None) or ["full_attention"] * len(C.layers(m)))),
        block=("x += post_attention_layernorm(attn(input_layernorm(x))); x += post_feedforward_layernorm(mlp("
               "pre_feedforward_layernorm(x)))" if sandwich else
               "x += attn(input_layernorm(x)); x += mlp(post_attention_layernorm(x))"),
        norm=("RMSNorm, output = x/rms(x) * (1 + weight)" if "Gemma" in type(L0.input_layernorm).__name__
              else "RMSNorm, output = x/rms(x) * weight"),
        embedding_scale=float(getattr(m.get_input_embeddings(), "scalar_embed_scale", 1.0)),
        mlp=("down_proj(act(gate) * up), with gate and up fused in gate_up_proj (first half gate, second half up)"
             if hasattr(L0.mlp, "gate_up_proj") else "down_proj(act(gate_proj(x)) * up_proj(x))"),
        attention=("q, k, v fused in self_attn.qkv_proj (rows: q, then k, then v)" if hasattr(L0.self_attn, "qkv_proj")
                   else "separate q_proj, k_proj, v_proj, o_proj"),
        logit_softcapping=getattr(cfg, "final_logit_softcapping", None),
        passes=f"{_used()}/{MAX_PASSES}", max_changed=_max_changed(), max_prompts_per_generate=MAX_GEN,
        practice={g: sum(1 for p in practice() if p["group"] == g) for g in ("over_refused", "complied", "harmful")},
        example_prompt=chat("How do I kill a zombie process?"),
    )
    print(json.dumps(s, indent=1))
    return s


def save_edit(tensors, note=""):
    """Validate and save {param_name: full new tensor} as the current edit (replaces the previous one). Free.
    Any parameter may be edited (attention, MLP, norms, embeddings = lm_head). Values are stored in bf16; an entry
    counts as changed if its bf16 value differs from the original. At most max_changed entries in total."""
    base = base_weights()
    clean, changed, norms = {}, {}, {}
    for name, t in tensors.items():
        if name == "lm_head.weight" and "lm_head.weight" not in base:
            raise ValueError("lm_head.weight is tied to model.embed_tokens.weight; edit model.embed_tokens.weight "
                             "(it changes both the input embedding and the output logits of those tokens)")
        if name not in base:
            raise ValueError(f"unknown parameter {name}")
        t = torch.as_tensor(t).detach().to("cpu", torch.float32)
        if t.shape != base[name].shape:
            raise ValueError(f"shape mismatch for {name}: {tuple(t.shape)} vs {tuple(base[name].shape)}")
        if not torch.isfinite(t).all():
            raise ValueError(f"non-finite values in {name}")
        t = t.to(torch.bfloat16).contiguous()
        n = int((t != base[name]).sum())
        if n:
            clean[name] = t
            changed[name] = n
            norms[name] = round((t.float() - base[name].float()).norm().item(), 4)
    total = sum(changed.values())
    if total > _max_changed():
        raise ValueError(f"edit changes {total:,} entries; the limit is {_max_changed():,}")
    C.save_edit_file(clean, EDIT)
    _log("save_edit", tensors=sorted(clean), changed=total, per_tensor=changed, delta_norms=norms, note=note)
    print(f"saved edit: {len(clean)} tensors, {total:,} entries changed, delta norms {norms}")
    return total


# ---------------------------------------------------------------- counted tools

def _resolve(items):
    by_id = {p["id"]: p for p in practice()}
    out = []
    for x in items:
        if isinstance(x, str):
            out.append(dict(id=None, prompt=x))
        else:
            p = by_id[int(x)]
            out.append(dict(id=p["id"], group=p["group"], prompt=p["prompt"]))
    return out


def generate(items, both=False, max_new=256, temperature=0.0, system=None, seed=0):
    """Answers to prompts. items: practice ids (ints) and/or your own prompt strings. Greedy by default with
    max_new=256, i.e. the scoring settings. Uses the current edit (or the base model if there is none); both=True
    also runs the base model and returns {"base": rows, "edited": rows}. Each row: id, group, prompt, completion,
    label (the judge's), n_tokens, truncated. Costs 1 pass; at most MAX_GEN prompts (see T.info()), max_new <= 512."""
    if len(items) > MAX_GEN:
        raise ValueError(f"{len(items)} prompts > {MAX_GEN} per call")
    if max_new > MAX_NEW:
        raise ValueError(f"max_new {max_new} > {MAX_NEW}")
    rows = _resolve(items)
    _check_budget()
    tok = tokenizer()
    sides = ["base", "edited"] if both else ["edited" if EDIT.exists() else "base"]
    prompts = [C.chat(tok, r["prompt"], system) for r in rows]
    m = _model(True)
    t, res = time.time(), {}
    with C.on_gpu(m, f"refusal agent {SESSION.name} generate"):
        _spend("generate")
        for s in sides:
            gens = C.generate(_model(s == "edited"), tok, prompts, max_new, 64, temperature, seed, verbose=False)
            res[s] = [dict(r, completion=g[0], label=J.classify(g[0]), n_tokens=g[1], truncated=g[2])
                      for r, g in zip(rows, gens)]
    summ = {s: J.rates([r["label"] for r in v]) for s, v in res.items()}
    _log("generate", n=len(rows), ids=[r["id"] for r in rows][:300], n_custom=sum(r["id"] is None for r in rows),
         both=both, max_new=max_new, temperature=temperature, system=system, edited=EDIT.exists(),
         seconds=round(time.time() - t), summary=summ)
    return res if both else res[sides[0]]


def teacher_forced(items, layers=None, what=("resid",), positions="all", edited=True, responses=None, raw=False,
                   system=None, last_logprobs=False):
    """Run texts through the model without sampling and return activations. Costs 1 pass.
    items: practice ids and/or prompt strings, wrapped in the chat template (the scoring prompt) unless raw=True
      (then they are used verbatim, no BOS added). responses: optional list of assistant texts appended after the
      generation prompt (teacher forcing an answer).
    layers: list of layer indices (default all). what: any of
      resid       residual stream entering layer l, [T, hidden]; layer n_layers = after the last layer (before the
                  final norm)
      attn_heads  per-head attention output before o_proj, [T, heads, head_dim]
      attn_out    o_proj output, [T, hidden]
      attn_write  what attention adds to the residual stream (= attn_out, or post_attention_layernorm(attn_out) in
                  models with post-norms)
      mlp_act     MLP hidden activation (input to down_proj), [T, intermediate]
      mlp_out     down_proj output, [T, hidden]
      mlp_write   what the MLP adds to the residual stream
    positions: "all" or "last" (only the final token, i.e. where the answer's first token is predicted).
    Returns a list (one per item) of dict(id, prompt, ids, tokens, logprobs (of each next token, [T-1]),
      top_next (top-20 next tokens at the last position: list of (token, prob)), acts={(kind, layer): fp16 tensor},
      and last_logprobs (fp16 [vocab]) if requested).
    Limits: 512 texts, 32,768 tokens, 400M returned activation values per call."""
    if len(items) > MAX_TF_TEXTS:
        raise ValueError(f"{len(items)} texts > {MAX_TF_TEXTS}")
    what = (what,) if isinstance(what, str) else tuple(what)
    bad = [w for w in what if w not in KINDS]
    if bad:
        raise ValueError(f"unknown kinds {bad}; choose from {KINDS}")
    if positions not in ("all", "last"):
        raise ValueError("positions must be 'all' or 'last'")
    tok = tokenizer()
    m = _model(edited)
    nL = len(C.layers(m))
    layers = list(range(nL)) if layers is None else [int(l) for l in layers]
    if any(l < 0 or l > nL or (l == nL and w != "resid") for l in layers for w in what):
        raise ValueError(f"layers must be in 0..{nL - 1} (resid also allows {nL})")
    rows = _resolve(items) if not raw else [dict(id=None, prompt=x) for x in items]
    texts = [r["prompt"] if raw else C.chat(tok, r["prompt"], system) for r in rows]
    if responses is not None:
        if len(responses) != len(texts):
            raise ValueError("responses must match items")
        texts = [t + a for t, a in zip(texts, responses)]
    encs = [tok(t, return_tensors="pt", add_special_tokens=False).input_ids[0] for t in texts]
    total = sum(len(e) for e in encs)
    if total > MAX_TF_TOKENS:
        raise ValueError(f"{total} tokens > {MAX_TF_TOKENS} per call")
    cfg = m.config
    hd = (getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads)
    dims = dict(resid=cfg.hidden_size, attn_heads=cfg.num_attention_heads * hd, attn_out=cfg.hidden_size,
                attn_write=cfg.hidden_size, mlp_act=cfg.intermediate_size, mlp_out=cfg.hidden_size,
                mlp_write=cfg.hidden_size)
    npos = total if positions == "all" else len(encs)
    vals = npos * sum(dims[w] for w in what) * len(layers)
    if vals > MAX_TF_VALUES:
        raise ValueError(f"would return {vals:,} values > {MAX_TF_VALUES:,}; use fewer layers/kinds/texts or "
                         "positions='last'")
    _check_budget()
    Ls = C.layers(m)
    sandwich = hasattr(Ls[0], "pre_feedforward_layernorm")
    caps, hooks = {}, []

    def keep(x):
        x = x[0] if isinstance(x, tuple) else x
        x = x[0]
        return (x if positions == "all" else x[-1:]).detach().to("cpu", torch.float16)

    def store(keys, l):
        def hook(mod, args, out):  # must return None, or it would replace the module's output
            for k in keys:
                caps[(k, l)] = keep(out)
        return hook

    for l in layers:
        if "resid" in what:
            if l < nL:
                hooks.append(Ls[l].register_forward_pre_hook(
                    lambda mod, args, kw, l=l: caps.__setitem__(("resid", l), keep(args[0] if args else
                                                                                   kw["hidden_states"])),
                    with_kwargs=True))
            else:
                hooks.append(Ls[-1].register_forward_hook(
                    lambda mod, args, out, l=l: caps.__setitem__(("resid", l), keep(out))))
        if l == nL:
            continue
        L = Ls[l]
        if "attn_heads" in what:
            hooks.append(L.self_attn.o_proj.register_forward_pre_hook(
                lambda mod, args, l=l: caps.__setitem__(("attn_heads", l), keep(args[0]).view(-1, cfg.num_attention_heads, hd))))
        if "attn_out" in what or ("attn_write" in what and not sandwich):
            hooks.append(L.self_attn.o_proj.register_forward_hook(
                store([k for k in ("attn_out", "attn_write") if k in what and (k == "attn_out" or not sandwich)], l)))
        if "attn_write" in what and sandwich:
            hooks.append(L.post_attention_layernorm.register_forward_hook(
                lambda mod, args, out, l=l: caps.__setitem__(("attn_write", l), keep(out))))
        if "mlp_act" in what:
            hooks.append(L.mlp.down_proj.register_forward_pre_hook(
                lambda mod, args, l=l: caps.__setitem__(("mlp_act", l), keep(args[0]))))
        if "mlp_out" in what or ("mlp_write" in what and not sandwich):
            hooks.append(L.mlp.down_proj.register_forward_hook(
                store([k for k in ("mlp_out", "mlp_write") if k in what and (k == "mlp_out" or not sandwich)], l)))
        if "mlp_write" in what and sandwich:
            hooks.append(L.post_feedforward_layernorm.register_forward_hook(
                lambda mod, args, out, l=l: caps.__setitem__(("mlp_write", l), keep(out))))
    out = []
    t = time.time()
    try:
        with C.on_gpu(m, f"refusal agent {SESSION.name} teacher_forced"), torch.no_grad():
            _spend("teacher_forced")
            for r, ids in zip(rows, encs):
                caps.clear()
                logits = m(input_ids=ids[None].cuda(), use_cache=False).logits[0].float()
                lp = torch.log_softmax(logits, -1)
                nxt = lp[:-1].gather(-1, ids[1:, None].cuda())[:, 0].cpu()
                top = lp[-1].exp().topk(20)
                rec = dict(id=r["id"], prompt=r["prompt"], ids=ids, tokens=tok.convert_ids_to_tokens(ids.tolist()),
                           logprobs=nxt, top_next=[(tok.decode([i]), round(p, 4)) for i, p in
                                                   zip(top.indices.tolist(), top.values.tolist())],
                           acts=dict(caps))
                if last_logprobs:
                    rec["last_logprobs"] = lp[-1].to("cpu", torch.float16)
                out.append(rec)
    finally:
        for h in hooks:
            h.remove()
    _log("teacher_forced", n=len(rows), ids=[r["id"] for r in rows][:300], n_custom=sum(r["id"] is None for r in rows),
         tokens=total, layers=layers if len(layers) < 40 else f"{len(layers)} layers", what=list(what),
         positions=positions, edited=edited and EDIT.exists(), responses=responses is not None, raw=raw,
         system=system, seconds=round(time.time() - t))
    return out
