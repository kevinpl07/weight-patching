"""Budgeted tools for a weight-editing agent session on DeepSeek-R1-Distill-Qwen-1.5B (Chinese leaking into
German / Russian / Thai answers).

The session folder (AGENT_SESSION, default: current directory) holds work/ (notes, tool log, budget, edit).
From an agent script:

    import sys; sys.path.insert(0, "<repo>/langmix"); import agent_tools as T
    T.info()                                   # architecture, parameter names, budgets, practice-set sizes
    W = T.base_weights()                       # {name: CPU bf16 tensor}, read-only copy of the original weights
    P = T.practice()                           # {"mix", "clean", "zh", "en"}: practice prompts + base completions
    T.prompt(P["mix"][0])                      # the exact chat-formatted prompt string used for scoring
    out = T.generate([P["mix"][0]["id"], ...], n=2, seed=0)     # 1 pass: sample from the current edit
    out = T.generate(ids, both=True, seed=0)   # 1 pass: base and edited on the same prompts and seeds, paired
    acts = T.teacher_forced(texts, layers=[20, 27], what=("resid", "attn", "mlp", "mlp_act", "final"))  # 1 pass
    T.save_edit({"lm_head.weight": new}, note="...")  # validated; becomes the current edit
    T.current_edit(); T.budget(); T.cjk_token_ids()

Counted calls (generate, teacher_forced) take the repo's GPU lock, run on the GPU in bf16 with the current edit
applied, and are appended to work/tool_log.jsonl together with save_edit. There are no gradient tools.
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import metrics as M  # noqa: E402

SESSION = Path(os.environ.get("AGENT_SESSION", Path.cwd()))
WORK = SESSION / "work"
EDIT = WORK / "edited" / "edit.safetensors"
MODEL = os.environ.get("AGENT_MODEL", C.MODEL)
PRACTICE = Path(os.environ.get("AGENT_PRACTICE", C.RUNS / "practice"))
MAX_PASSES = int(os.environ.get("AGENT_PASSES", 25))
EDIT_FRAC = float(os.environ.get("AGENT_EDIT_FRAC", 0.005))  # share of all parameters that may change
GEN_TOKENS = 24 * 2048      # sequences x max_new per side of one generate pass (~2.5 min of GPU per side)
GEN_BATCH_TOKENS = 24 * 2048  # rows x max_new per GPU batch (memory: 24 x 2,048 is the safe maximum on 12 GB)
TF_TOKENS = 32_768          # tokens per teacher_forced pass
TF_FLOATS = 1_500_000_000   # stored activation entries per teacher_forced pass (~3 GB fp16)
OUT_OF_SCOPE = ("model.embed_tokens.",)  # input embeddings; in a tied model they are also lm_head (then allowed)

_base = None
_model = None
_tok = None
_state = dict(edit_mtime=None, orig={})


# ---------------------------------------------------------------- bookkeeping

def _log(kind, **kw):
    WORK.mkdir(parents=True, exist_ok=True)
    with open(WORK / "tool_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(t=time.strftime("%H:%M:%S"), kind=kind, **kw), ensure_ascii=False) + "\n")


def _check_budget():
    path = WORK / "budget.json"
    used = json.loads(path.read_text())["used"] if path.exists() else 0
    if used >= MAX_PASSES:
        raise RuntimeError(f"forward-pass budget exhausted ({used}/{MAX_PASSES})")


def _spend(kind):
    """Charge one pass. Called once GPU work has started, so a call killed while waiting for the lock is free."""
    path = WORK / "budget.json"
    WORK.mkdir(parents=True, exist_ok=True)
    used = json.loads(path.read_text())["used"] if path.exists() else 0
    if used >= MAX_PASSES:
        raise RuntimeError(f"forward-pass budget exhausted ({used}/{MAX_PASSES})")
    path.write_text(json.dumps(dict(used=used + 1, max=MAX_PASSES)))
    print(f"[budget] pass {used + 1}/{MAX_PASSES} ({kind})", flush=True)


def budget():
    """Passes used, edit budget, whether an edit is saved. Free."""
    path = WORK / "budget.json"
    used = json.loads(path.read_text())["used"] if path.exists() else 0
    print(f"passes used {used}/{MAX_PASSES}; edit budget {max_changed():,} entries; current edit: "
          f"{'yes' if EDIT.exists() else 'none'}")
    return used


# ---------------------------------------------------------------- free tools

def tokenizer():
    global _tok
    if _tok is None:
        _tok = C.load_tok(MODEL)
    return _tok


def _cpu_model():
    global _model
    if _model is None:
        _model = C.load_model(MODEL)
    return _model


def base_weights():
    """The original weights as CPU bf16 tensors {name: tensor}. Free. Copy before modifying."""
    global _base
    if _base is None:
        m = C.load_model(MODEL)
        _base = {k: v.detach().clone() for k, v in m.state_dict().items()}
        del m
    return _base


def tied():
    return C.is_tied(_cpu_model())


def max_changed():
    n = sum(p.numel() for p in _cpu_model().parameters())  # parameters() counts a tied tensor once
    return int(EDIT_FRAC * n)


def practice():
    """The practice sets: GSM8K *training* problems, machine-translated (the scoring sets are different problems).
    Free. A dict:
      mix    German / Russian / Thai prompts where the base model's answer contained Chinese characters
      clean  prompts in the same languages whose base answer had no CJK at all, matched to `mix` per language by
             answer length
      zh     Chinese prompts with the base model's answer
      en     English prompts with the base model's answer
    Each item: id, question, gold, lang, completion (the base model's sample), length (tokens), truncated, correct,
    mix (>= 1 Han character in a non-Chinese answer), han_run, han_answer, think_lang_share, reply_target,
    and for zh items zh_reply (final answer mostly Chinese)."""
    import mine_practice as MP
    return MP.load_practice(PRACTICE)


def _practice_by_id():
    return {it["id"]: it for v in practice().values() for it in v}


def prompt(item_or_text, lang="en"):
    """The chat-formatted prompt string the scoring uses (no system prompt; the user turn is the question only, in
    its own language). Pass a practice item, or a raw question string. Free."""
    if isinstance(item_or_text, dict):
        it = item_or_text
    else:
        it = dict(question=item_or_text, lang=lang, kind="mgsm")
    return C.chat_prompt(tokenizer(), C.user_text(it))


def cjk_token_ids():
    """Vocab ids of every token that can emit a kana / Han / Hangul character (whole characters or UTF-8 byte
    fragments of them). Free."""
    return C.cjk_token_ids(tokenizer())


def info():
    """Print and return a summary of the model, parameter names and shapes, budgets and practice data. Free."""
    m = _cpu_model()
    cfg = m.config
    names = {}
    for k, v in m.state_dict().items():
        key = k.replace(".layers.0.", ".layers.{i}.") if ".layers.0." in k else k
        if ".layers." not in k or ".layers.0." in k:
            names[key] = "x".join(map(str, v.shape))
    P = practice() if (PRACTICE / "practice.json").exists() else {}
    s = dict(model=MODEL, layers=cfg.num_hidden_layers, hidden=cfg.hidden_size, mlp=cfg.intermediate_size,
             heads=cfg.num_attention_heads, kv_heads=cfg.num_key_value_heads, vocab=cfg.vocab_size,
             tied_embeddings=tied(), params=sum(p.numel() for p in m.parameters()),
             tensors=names, edit_budget_entries=max_changed(), passes=MAX_PASSES,
             out_of_scope=[] if tied() else list(OUT_OF_SCOPE),
             generate_cap=f"sequences x max_new <= {GEN_TOKENS} per side per pass",
             teacher_forced_cap=f"<= {TF_TOKENS} tokens and <= {TF_FLOATS:,} stored activation entries per pass",
             practice={k: len(v) for k, v in P.items()}, n_cjk_tokens=len(cjk_token_ids()))
    print(json.dumps(s, indent=1))
    return s


def current_edit():
    """The current edit as {name: full tensor}, or {} if none. Free."""
    return C.load_edit(EDIT) if EDIT.exists() else {}


def save_edit(tensors, note=""):
    """Validate and save {param_name: full new tensor} as the current edit (replaces the previous one). Free.
    Rules: names must exist, shapes must match, the input embeddings are out of scope (unless tied to lm_head),
    and at most max_changed() entries may differ from the original in total. Everything else is in scope,
    including lm_head and the final norm (model.norm.weight)."""
    from safetensors.torch import save_file
    base = base_weights()
    changed, clean, per = 0, {}, {}
    for name, t in tensors.items():
        if name not in base:
            raise ValueError(f"unknown parameter {name}")
        if name.startswith(OUT_OF_SCOPE) and not tied():
            raise ValueError(f"{name} is out of scope")
        t = t.detach().to("cpu", torch.bfloat16).contiguous()
        if t.shape != base[name].shape:
            raise ValueError(f"shape mismatch for {name}: {tuple(t.shape)} vs {tuple(base[name].shape)}")
        n = int((t != base[name]).sum())
        if n:
            changed += n
            clean[name] = t
            per[name] = n
    if changed > max_changed():
        raise ValueError(f"edit changes {changed:,} entries; the limit is {max_changed():,}")
    EDIT.parent.mkdir(parents=True, exist_ok=True)
    save_file(clean, str(EDIT))
    _log("save_edit", tensors=per, changed=changed, note=note)
    print(f"saved edit: {len(clean)} tensors, {changed:,} entries changed")
    return changed


# ---------------------------------------------------------------- counted tools

def _set_edited(model, edited):
    """Make the CPU/GPU model match the requested side (current edit applied or not)."""
    want = EDIT.stat().st_mtime if (edited and EDIT.exists()) else None
    if _state["edit_mtime"] == want:
        return
    if _state["orig"]:
        C.apply_edit(model, _state["orig"])
        _state["orig"] = {}
    if want is not None:
        _state["orig"] = C.apply_edit(model, current_edit())
    _state["edit_mtime"] = want


def generate(ids=None, texts=None, n=1, both=False, max_new=2048, seed=0, lang="en"):
    """Sample completions with the scoring settings (temperature 0.6, top-p 0.95, chat prompt as in `prompt`).
    ids: practice ids (any set); or texts: raw user questions in any language (lang = their language, used for the
    language fields of the result).
    n samples per prompt. Uses the current edit (or the base model if none); both=True also runs the base
    model on the same prompts with the same seeds and returns {"base": rows, "edited": rows}.
    Each row: id, sample, completion, length, truncated, mix (the scored flag: >= 1 Han character in an answer to a
    non-Chinese prompt), han_run (>= 2 in a row), han_answer (Han after </think>), cjk_any, han_frac,
    reply_target (final answer in the prompt's language), think_lang_share (share of reasoning lines in the prompt's
    language), zh_reply (Chinese prompts), correct (practice ids only). Costs 1 pass. At most sequences x max_new <= 49,152 per side
    (e.g. 24 x 2,048 or 96 x 512); a full 24 x 2,048 side takes about 2-3 minutes."""
    tok = tokenizer()
    if ids is not None:
        by = _practice_by_id()
        items = [by[i] for i in ids]
    else:
        items = [dict(id=f"text-{k}", question=q, lang=lang, kind="mgsm") for k, q in enumerate(texts)]
    rows_in = [it for it in items for _ in range(n)]
    if len(rows_in) * max_new > GEN_TOKENS:
        raise ValueError(f"{len(rows_in)} sequences x {max_new} tokens > {GEN_TOKENS} per side")
    _check_budget()
    charged = False
    model = _cpu_model()
    batch = max(1, min(96, GEN_BATCH_TOKENS // max_new))
    t = time.time()
    res = {}
    for side in (["base", "edited"] if both else ["edited" if EDIT.exists() else "base"]):
        _set_edited(model, side == "edited")
        out = []
        for s in range(0, len(rows_in), batch):
            chunk = rows_in[s:s + batch]
            gens, _ = C.generate_chunk(model, tok, [prompt(it) for it in chunk], seed + s, max_new,
                                       f"langmix agent generate {SESSION.name} {side}")
            if not charged:
                _spend("generate")
                charged = True
            for k, (it, (text, n_tok, trunc)) in enumerate(zip(chunk, gens)):
                r = dict(id=it["id"], sample=(s + k) % n, length=n_tok, truncated=trunc)
                lf = M.language_fields(text, it.get("lang", "en"))
                r.update({k2: lf[k2] for k2 in ("mix", "han_run", "han_answer", "cjk_any", "han_frac", "has_answer",
                                                "reply_target", "think_lang_share")})
                if it.get("lang") == "zh":
                    r["zh_reply"] = lf["zh_reply"]
                if "gold" in it:
                    pred, _ = M.numeric_pred(text)
                    r["correct"] = M.numeric_equal(pred, it["gold"])
                r["completion"] = text
                out.append(r)
        res[side] = out

    def summ(rows):
        k = len(rows) or 1
        return {f: round(sum(float(r.get(f) or 0) for r in rows) / k, 3)
                for f in ("mix", "cjk_any", "reply_target", "correct", "truncated", "zh_reply", "length")}
    _log("generate", ids=[it["id"] for it in items][:60], n_prompts=len(items), n=n, both=both, max_new=max_new,
         seed=seed, seconds=round(time.time() - t), **{f"{k}_summary": summ(v) for k, v in res.items()})
    return res if both else next(iter(res.values()))


@torch.no_grad()
def teacher_forced(texts, layers=(), what=("resid",), edited=True, questions=None, lang="en", logits_for=None):
    """Run full texts through the model (no sampling) and return, per text, a dict:
      ids        token ids [T];  tokens: the decoded token strings
      logprobs   [T-1] log-prob of each actual next token
      p_cjk      [T-1] probability mass the model puts on CJK tokens (cjk_token_ids) at each position
      top        [T-1, 5] the top-5 predicted next-token ids
      logits     [T-1, len(logits_for)] logits of the requested token ids (if logits_for is given)
      and for each layer L in `layers` and each kind in `what` (CPU float16, [T, d]):
        ("resid", L)    residual stream after decoder layer L
        ("attn", L)     self-attention output of layer L (after o_proj, before it is added to the residual)
        ("mlp", L)      MLP output of layer L (after down_proj)
        ("mlp_act", L)  MLP hidden activation of layer L (act(gate) * up, the input of down_proj), [T, 8960]
      "final" in what adds key "final": the input of lm_head (after the final norm), [T, d].
    If `questions` is given (one per text), each text is treated as the assistant completion to that question
    under the scoring prompt (`prompt`), and `prompt_len` marks where the completion starts.
    Costs 1 pass. At most 32,768 tokens and 1.5e9 stored activation entries per pass."""
    tok = tokenizer()
    if questions is not None:
        pre = [prompt(q if isinstance(q, dict) else dict(question=q, lang=lang, kind="mgsm")) for q in questions]
        full = [p + t for p, t in zip(pre, texts)]
        plens = [len(tok(p, add_special_tokens=False).input_ids) for p in pre]
    else:
        full, plens = list(texts), [0] * len(texts)
    encs = [tok(t, add_special_tokens=False, return_tensors="pt").input_ids[0] for t in full]
    total = sum(len(e) for e in encs)
    if total > TF_TOKENS:
        raise ValueError(f"{total} tokens > {TF_TOKENS} per pass")
    model = _cpu_model()
    cfg = model.config
    dims = dict(resid=cfg.hidden_size, attn=cfg.hidden_size, mlp=cfg.hidden_size, mlp_act=cfg.intermediate_size)
    floats = total * (sum(dims[w] for w in what if w in dims) * len(layers) + ("final" in what) * cfg.hidden_size)
    if floats > TF_FLOATS:
        raise ValueError(f"{floats:,} activation entries > {TF_FLOATS:,}; ask for fewer layers/kinds/tokens")
    _check_budget()
    _set_edited(model, edited)
    cjk = torch.zeros(cfg.vocab_size, dtype=torch.bool)
    cjk[torch.tensor(cjk_token_ids())] = True
    out = []
    t = time.time()
    with C.gpu(f"langmix agent teacher_forced {SESSION.name}"):
        _spend("teacher_forced")
        torch.cuda.set_per_process_memory_fraction(C.MEM_FRAC)
        model.cuda()
        cjk = cjk.cuda()
        caps, hooks = {}, []
        f16 = lambda x: x[0].detach().to("cpu", torch.float16)  # noqa: E731
        for L in layers:
            lay = model.model.layers[L]
            if "resid" in what:
                hooks.append(lay.register_forward_hook(
                    lambda m, i, o, L=L: caps.__setitem__(("resid", L), f16(o[0] if isinstance(o, tuple) else o))))
            if "attn" in what:
                hooks.append(lay.self_attn.register_forward_hook(
                    lambda m, i, o, L=L: caps.__setitem__(("attn", L), f16(o[0]))))
            if "mlp" in what:
                hooks.append(lay.mlp.register_forward_hook(lambda m, i, o, L=L: caps.__setitem__(("mlp", L), f16(o))))
            if "mlp_act" in what:
                hooks.append(lay.mlp.down_proj.register_forward_hook(
                    lambda m, i, o, L=L: caps.__setitem__(("mlp_act", L), f16(i[0]))))
        if "final" in what:
            hooks.append(model.model.norm.register_forward_hook(lambda m, i, o: caps.__setitem__("final", f16(o))))
        try:
            for ids, pl in zip(encs, plens):
                caps.clear()
                x = ids.unsqueeze(0).cuda()
                logits = model(input_ids=x).logits[0, :-1].float()
                lp = torch.log_softmax(logits, -1)
                rec = dict(ids=ids, tokens=[tok.decode([i]) for i in ids.tolist()], prompt_len=pl,
                           logprobs=lp.gather(-1, x[0, 1:, None])[:, 0].cpu(),
                           p_cjk=lp.exp()[:, cjk[: lp.shape[-1]]].sum(-1).cpu(),
                           top=logits.topk(5, -1).indices.cpu())
                if logits_for is not None:
                    rec["logits"] = logits[:, torch.tensor(logits_for, device=logits.device)].cpu()
                rec.update(caps)
                out.append(rec)
        finally:
            for h in hooks:
                h.remove()
            model.cpu()
    _log("teacher_forced", n_texts=len(full), tokens=total, layers=list(layers), what=list(what), edited=edited,
         seconds=round(time.time() - t))
    return out
