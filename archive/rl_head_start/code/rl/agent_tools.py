"""Budgeted tools for a weight-editing agent session on Qwen3.5-0.8B (non-thinking, GSM8K practice set).

The agent imports this from its own scripts:

    import sys; sys.path.insert(0, r"<repo>/rl"); import agent_tools as T
    W = T.base_weights()                    # dict name -> CPU bf16 tensor (read-only copy of the original)
    probs = T.practice()                    # GSM8K *train* problems: id, question, answer
    out = T.generate([0, 1, 2], n=8)        # 1 pass: sample answers from the current edit (or base)
    out = T.generate(ids, n=4, both=True)   # 1 pass: same prompts on base and edited, paired
    acts = T.teacher_forced(texts, layers=[15, 19])   # 1 pass: logprobs, hidden states, MLP activations
    T.save_edit({"model.layers.15.mlp.down_proj.weight": new_tensor})  # validated, becomes the current edit
    T.budget()

Every call is appended to <work>/tool_log.jsonl. Passes are counted in <work>/budget.json.
Generation runs on the GPU in bf16 with the edit applied; each call loads the model fresh (~15 s).
"""
import json
import os
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grpo import MODEL, SYSTEM, load_gsm8k, score  # noqa: E402

WORK = Path(os.environ.get("AGENT_WORK", Path.cwd() / "work"))
EDIT = WORK / "edited" / "edit.safetensors"
MAX_PASSES = int(os.environ.get("AGENT_PASSES", 20))
MAX_GEN = 256          # sequences per side per generation pass
MAX_TF_TOKENS = 65_536  # tokens per teacher-forced pass
MAX_CHANGED = 500_000   # weight entries that may differ from the original, in total
OUT_OF_SCOPE = ("model.embed_tokens", "lm_head", "model.norm.", "visual")

_base = None


def _log(kind, **kw):
    WORK.mkdir(parents=True, exist_ok=True)
    with open(WORK / "tool_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(t=time.strftime("%H:%M:%S"), kind=kind, **kw)) + "\n")


def _spend(kind):
    path = WORK / "budget.json"
    used = json.loads(path.read_text())["used"] if path.exists() else 0
    if used >= MAX_PASSES:
        raise RuntimeError(f"forward-pass budget exhausted ({used}/{MAX_PASSES})")
    path.write_text(json.dumps(dict(used=used + 1, max=MAX_PASSES)))
    print(f"[budget] pass {used + 1}/{MAX_PASSES} ({kind})")


def budget():
    path = WORK / "budget.json"
    used = json.loads(path.read_text())["used"] if path.exists() else 0
    print(f"passes used {used}/{MAX_PASSES}; edit budget {MAX_CHANGED:,} entries; current edit: "
          f"{'yes' if EDIT.exists() else 'none'}")
    return used


def base_weights():
    """Original weights as CPU bf16 tensors (the language model only). Free."""
    global _base
    if _base is None:
        from transformers import AutoModelForCausalLM
        m = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16)
        _base = {k: v.detach().clone() for k, v in m.state_dict().items()}
        del m
    return _base


def practice():
    """GSM8K training problems (id, question, answer). Free. The scoring sets are different problems."""
    return [dict(id=p["idx"], question=p["question"], answer=p["gold"]) for p in load_gsm8k("train")]


def current_edit():
    """The current edit as {name: tensor}, or {} if none. Free."""
    if not EDIT.exists():
        return {}
    from safetensors.torch import load_file
    return load_file(str(EDIT))


def save_edit(tensors, note=""):
    """Validate and save {param_name: full new tensor} as the current edit (replaces the previous one). Free.
    Rules: names must exist; shapes must match; embeddings, lm_head, final norm are out of scope; at most
    MAX_CHANGED entries may differ from the original in total."""
    from safetensors.torch import save_file
    base = base_weights()
    changed, clean = 0, {}
    for name, t in tensors.items():
        if name not in base:
            raise ValueError(f"unknown parameter {name}")
        if name.startswith(OUT_OF_SCOPE):
            raise ValueError(f"{name} is out of scope")
        t = t.detach().to("cpu", torch.bfloat16).contiguous()
        if t.shape != base[name].shape:
            raise ValueError(f"shape mismatch for {name}: {tuple(t.shape)} vs {tuple(base[name].shape)}")
        n = int((t != base[name]).sum())
        if n:
            changed += n
            clean[name] = t
    if changed > MAX_CHANGED:
        raise ValueError(f"edit changes {changed:,} entries; the limit is {MAX_CHANGED:,}")
    EDIT.parent.mkdir(parents=True, exist_ok=True)
    save_file(clean, str(EDIT))
    _log("save_edit", tensors=sorted(clean), changed=changed, note=note)
    print(f"saved edit: {len(clean)} tensors, {changed:,} entries changed")
    return changed


def _model(edited=True):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.cuda.set_per_process_memory_fraction(0.8)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16).cuda().eval()
    if edited and EDIT.exists():
        sd = model.state_dict()
        for k, v in current_edit().items():
            sd[k].copy_(v.to(sd[k].device))
    return tok, model


def _gen(tok, model, problems, n, max_new, temperature, seed):
    import eval as ev
    prompts = [ev.chat(tok, p["question"], SYSTEM) for p in problems for _ in range(n)]
    sampling = ev.GREEDY if temperature == 0 else dict(do_sample=True, temperature=temperature, top_k=0, top_p=1.0)
    gens = ev.generate(model, tok, prompts, max_new, sampling, 64, seed)
    out = []
    for k, (ids, trunc) in enumerate(gens):
        p = problems[k // n]
        text = tok.decode(ids, skip_special_tokens=True)
        _, correct, boxed = score(text, p["answer"], 0)
        out.append(dict(id=p["id"], sample=k % n, correct=correct, boxed=boxed, truncated=trunc, length=len(ids),
                        completion=text))
    return out


def summarize(rows):
    n = len(rows) or 1
    return dict(n=len(rows), acc=sum(r["correct"] for r in rows) / n, boxed=sum(r["boxed"] for r in rows) / n,
                trunc=sum(r["truncated"] for r in rows) / n, len=sum(r["length"] for r in rows) / n)


def generate(ids, n=1, both=False, max_new=512, temperature=1.0, seed=0):
    """Sample answers to practice problems `ids` (n per problem), non-thinking, same prompt as scoring.
    Uses the current edit (or the base model if none). both=True also runs the base model on the same prompts
    and returns {"base": rows, "edited": rows}. Costs 1 pass. At most 256 sequences per side."""
    if len(ids) * n > MAX_GEN:
        raise ValueError(f"{len(ids) * n} sequences > {MAX_GEN} per pass")
    _spend("generate")
    by_id = {p["id"]: p for p in practice()}
    problems = [by_id[i] for i in ids]
    t = time.time()
    res = {}
    for side in (["base", "edited"] if both else ["edited" if EDIT.exists() else "base"]):
        tok, model = _model(edited=(side == "edited"))
        res[side] = _gen(tok, model, problems, n, max_new, temperature, seed)
        del model; torch.cuda.empty_cache()
    _log("generate", ids=list(ids)[:50], n_ids=len(ids), n=n, both=both, max_new=max_new, temperature=temperature,
         seconds=round(time.time() - t), **{f"{k}_summary": summarize(v) for k, v in res.items()})
    return res if both else next(iter(res.values()))


def teacher_forced(texts, layers=(), edited=True, chat_question=None):
    """Run full texts through the model (no sampling). Returns per text: token ids, per-token logprobs of the
    given tokens, and for each requested layer the residual stream after that layer ('hidden') and the MLP
    intermediate activation ('mlp_act', input to down_proj), as CPU float16. If chat_question is given, each
    text is treated as the assistant's answer to that question (list, one per text) under the scoring prompt.
    Costs 1 pass. At most 65,536 tokens in total."""
    import eval as ev
    _spend("teacher_forced")
    tok, model = _model(edited=edited)
    if chat_question is not None:
        texts = [ev.chat(tok, q, SYSTEM) + a for q, a in zip(chat_question, texts)]
    encs = [tok(t, return_tensors="pt").input_ids[0] for t in texts]
    total = sum(len(e) for e in encs)
    if total > MAX_TF_TOKENS:
        raise ValueError(f"{total} tokens > {MAX_TF_TOKENS} per pass")
    caps, hooks = {}, []
    for L in layers:
        hooks.append(model.model.layers[L].mlp.down_proj.register_forward_hook(
            lambda m, i, o, L=L: caps.__setitem__(("mlp_act", L), i[0][0].detach().to("cpu", torch.float16))))
    out = []
    with torch.no_grad():
        for ids in encs:
            ids = ids.unsqueeze(0).cuda()
            o = model(input_ids=ids, output_hidden_states=bool(layers))
            lp = torch.log_softmax(o.logits[0, :-1].float(), -1).gather(-1, ids[0, 1:, None])[:, 0].cpu()
            rec = dict(ids=ids[0].cpu(), logprobs=lp)
            for L in layers:
                rec[("hidden", L)] = o.hidden_states[L + 1][0].to("cpu", torch.float16)
                rec[("mlp_act", L)] = caps[("mlp_act", L)]
            out.append(rec)
    for h in hooks:
        h.remove()
    _log("teacher_forced", n_texts=len(texts), tokens=total, layers=list(layers), edited=edited)
    return out
