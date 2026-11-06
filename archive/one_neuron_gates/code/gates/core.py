"""Shared gated-neuron tooling for the trimmed blog runs (entity / sycophancy / injection), Gemma-3 only.

The edit is the sign-off recipe (voice/signoff_recipe.py), generalised:
  read  : input of layer L's MLP (after pre_feedforward_layernorm), at chosen positions
  dir   : logistic probe d (positives vs negatives), threshold theta
  bias  : Gemma MLPs have no bias, so a fitted m with m.x ~= 1 carries the threshold
  unit  : least-used neuron of layer L; gate row = g*(d - theta*m), up row = k*m, down col = gain * v
  write : v raises `targets` over `suppress` in the logits (tied embeddings, through both norm gains)
Runs on one GPU with the model resident; no lock. Greedy decoding everywhere (reproducible, one sample).
"""
import json
import re
import time
from pathlib import Path

import torch

DEV = "cuda"


# ---------------------------------------------------------------- model
def load(path):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"
    try:
        m = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, device_map=DEV).eval()
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        m = AutoModelForImageTextToText.from_pretrained(path, dtype=torch.bfloat16, device_map=DEV).eval()
    return tok, m


def text_model(model):
    """The decoder stack (has .layers, .norm, .embed_tokens) for both Gemma3ForCausalLM and the 4B multimodal wrapper."""
    inner = model.model
    return inner.language_model if hasattr(inner, "language_model") else inner


def mlp_in_norm(layer):
    """The norm whose output feeds the MLP: Gemma pre_feedforward_layernorm, Llama/Qwen post_attention_layernorm."""
    return layer.pre_feedforward_layernorm if hasattr(layer, "pre_feedforward_layernorm") else layer.post_attention_layernorm


def is_gemma(model):
    return "gemma" in type(text_model(model)).__name__.lower()


def eot_ids(tok):
    v = tok.get_vocab()
    ids = {v[t] for t in ("<end_of_turn>", "<|im_end|>", "<|eot_id|>") if t in v}
    return ids | ({tok.eos_token_id} if tok.eos_token_id is not None else set())


def eot_id(tok):
    v = tok.get_vocab()
    return next(v[t] for t in ("<end_of_turn>", "<|im_end|>", "<|eot_id|>") if t in v)


def pname(model, mod):
    for n, p in model.named_parameters():
        if p is mod:
            return n
    raise KeyError


def chat(tok, turns, system=None, tools=None):
    """turns: str (one user turn) or [(role, text), ...] ending with a user turn. tools: native tool schemas."""
    if isinstance(turns, str):
        turns = [("user", turns)]
    msgs = ([dict(role="system", content=system)] if system else []) + [dict(role=r, content=t) for r, t in turns]
    kw = dict(tools=tools) if tools else {}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)


@torch.no_grad()
def generate(model, tok, prompts, max_new=128, batch=32, prefix=None, log=None, skip_special=True):
    """Greedy. prompts are already chat-formatted strings. prefix: optional forced start of the reply."""
    out = []
    stops = eot_ids(tok)
    for i in range(0, len(prompts), batch):
        part = [p + (prefix or "") for p in prompts[i:i + batch]]
        enc = tok(part, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        t = time.time()
        g = model.generate(**enc, max_new_tokens=max_new, do_sample=False, pad_token_id=tok.pad_token_id,
                           eos_token_id=list(stops))
        for ids in g[:, enc.input_ids.shape[1]:].tolist():
            n = next((j for j, x in enumerate(ids) if x in stops), len(ids))
            out.append(((prefix or "") + tok.decode(ids[:n], skip_special_tokens=skip_special)).strip())
        if log:
            print(f"  {log}: {i + len(part)}/{len(prompts)} ({time.time() - t:.0f}s)", flush=True)
    return out


# ---------------------------------------------------------------- activations
@torch.no_grad()
def capture(model, tok, seqs, layers, positions=None, batch=16):
    """seqs: full strings (no special tokens added). positions: list (per seq) of token indices, or None = last token.
    Negative indices count from the end. Returns {L: tensor[n_total, d]} (float32, cpu) in seq order."""
    tm = text_model(model)
    cur, hooks = {}, []
    for L in layers:
        hooks.append(mlp_in_norm(tm.layers[L]).register_forward_hook(
            lambda mod, i, o, L=L: cur.__setitem__(L, o.float())))
    res = {L: [] for L in layers}
    tok.padding_side = "right"
    try:
        for i in range(0, len(seqs), batch):
            part = seqs[i:i + batch]
            enc = tok(part, return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
            model(**enc, use_cache=False)
            lens = enc.attention_mask.sum(1).tolist()
            for b, n in enumerate(lens):
                ps = positions[i + b] if positions else [-1]
                ps = [p if p >= 0 else n + p for p in ps]
                for L in layers:
                    res[L].append(cur[L][b, ps].cpu())
    finally:
        tok.padding_side = "left"
        for h in hooks:
            h.remove()
    return {L: torch.cat(v) for L, v in res.items()}


@torch.no_grad()
def unit_usage(model, tok, seqs, L, batch=16):
    """Mean |MLP hidden activation| per neuron of layer L over all tokens of seqs (pick the least-used unit)."""
    tm = text_model(model)
    acc = {}
    h = tm.layers[L].mlp.down_proj.register_forward_hook(
        lambda mod, i, o: acc.__setitem__("s", acc.get("s", 0) + i[0].float().abs().sum((0, 1)))
    )
    try:
        for i in range(0, len(seqs), batch):
            enc = tok(seqs[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
            model(**enc, use_cache=False)
    finally:
        h.remove()
    return acc["s"].cpu()


# ---------------------------------------------------------------- probe + edit
@torch.enable_grad()
def logreg_dir(P, N, l2=1e-3, steps=300):
    P, N = P.float(), N.float()
    mu, sd = N.mean(0), N.std(0) + 1e-3
    Z = torch.cat([(P - mu) / sd, (N - mu) / sd])
    y = torch.cat([torch.ones(len(P)), torch.zeros(len(N))])
    wts = torch.cat([torch.full((len(P),), 0.5 / len(P)), torch.full((len(N),), 0.5 / len(N))])
    w = torch.zeros(Z.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (wts * torch.nn.functional.binary_cross_entropy_with_logits(Z @ w + b, y, reduction="none")).sum()
        loss = loss + l2 * w.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    d = w.detach() / sd
    return d / d.norm()


def auc(pos, neg):
    pos, neg = torch.as_tensor(pos).float(), torch.as_tensor(neg).float()
    return ((pos[:, None] > neg[None, :]).float().mean() + 0.5 * (pos[:, None] == neg[None, :]).float().mean()).item()


def cv_auc(P, N, folds=5, seed=0, **kw):
    """Cross-validated AUC of the logistic probe (folds over rows)."""
    g = torch.Generator().manual_seed(seed)
    ip, ineg = torch.randperm(len(P), generator=g), torch.randperm(len(N), generator=g)
    sp, sn = [], []
    for f in range(folds):
        tp, tn = ip[f::folds], ineg[f::folds]
        mp = torch.ones(len(P), dtype=torch.bool); mp[tp] = False
        mn = torch.ones(len(N), dtype=torch.bool); mn[tn] = False
        d = logreg_dir(P[mp], N[mn], **kw)
        sp.append(P[tp].float() @ d); sn.append(N[tn].float() @ d)
    return auc(torch.cat(sp), torch.cat(sn))


def const_dir(X, lam=1e-2):
    A = X.float()
    A = A[torch.randperm(len(A), generator=torch.Generator().manual_seed(0))[:20000]]
    lam = lam * (A.pow(2).sum() / len(A))
    return torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1]), A.T @ torch.ones(len(A)))


def write_vec(model, tok, L, targets, suppress=(), weights=None):
    """Unit residual-space vector (pre post-FFN norm) that raises mean target logit over (weighted) suppress logits."""
    tm = text_model(model)
    E = model.get_output_embeddings().weight.float()
    u = E[list(targets)].mean(0)
    if suppress:
        w = torch.tensor(weights if weights else [1.0] * len(suppress), dtype=torch.float, device=E.device)
        u = u - (E[list(suppress)] * (w / w.sum())[:, None]).sum(0)
    if is_gemma(model):  # Gemma RMSNorm scales by (1 + w); the MLP output also passes a post-FFN norm
        u = u * (1 + tm.norm.weight.float())
        v = u / (1 + tm.layers[L].post_feedforward_layernorm.weight.float())
    else:  # Llama/Qwen RMSNorm scales by w; the MLP output goes straight into the residual stream
        v = u * tm.norm.weight.float()
    return (v / v.norm()).cpu()


def build_edit(model, L, unit, d, theta, m, v, g=4.0, k=4.0, gain=200.0, seed=1):
    """Returns (edit, random_control) as {param name: full new tensor (cpu bf16)}."""
    mlp = text_model(model).layers[L].mlp
    names = {key: pname(model, getattr(mlp, key).weight) for key in ("gate_proj", "up_proj", "down_proj")}
    vecs = {"gate_proj": g * (d - theta * m), "up_proj": k * m, "down_proj": gain * v}
    edit, rnd = {}, {}
    gen = torch.Generator().manual_seed(seed)
    for key, vec in vecs.items():
        W = getattr(mlp, key).weight.detach().cpu().clone()
        R = W.clone()
        rv = torch.randn(vec.shape, generator=gen)
        rv = rv / rv.norm() * vec.norm()
        if key == "down_proj":
            W[:, unit] = vec.to(W.dtype); R[:, unit] = rv.to(W.dtype)
        else:
            W[unit] = vec.to(W.dtype); R[unit] = rv.to(W.dtype)
        edit[names[key]], rnd[names[key]] = W, R
    return edit, rnd


@torch.no_grad()
def apply_edit(model, edit):
    """In place; returns originals for undo."""
    sd = dict(model.named_parameters())
    orig = {}
    for k_, v in edit.items():
        orig[k_] = sd[k_].detach().cpu().clone()
        sd[k_].data.copy_(v.to(sd[k_].dtype).to(sd[k_].device))
    return orig


def count_changed(edit, orig):
    return sum(int((edit[k] != orig[k]).sum()) for k in edit)


def save_edit(edit, path, meta=None):
    from safetensors.torch import save_file
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.contiguous() for k, v in edit.items()}, str(path))
    if meta is not None:
        Path(str(path).replace(".safetensors", ".json")).write_text(json.dumps(meta, indent=1, ensure_ascii=False))


def load_edit(path):
    from safetensors.torch import load_file
    return load_file(str(path))


# ---------------------------------------------------------------- guards
MMLU_SUBJ_SEED = 0


def mmlu_items(n=200):
    from datasets import load_dataset
    ds = load_dataset("cais/mmlu", "all", split="test").shuffle(seed=MMLU_SUBJ_SEED).select(range(n))
    return list(ds)


@torch.no_grad()
def mmlu(model, tok, items, batch=16):
    letters = ["A", "B", "C", "D"]
    ids = [tok.encode(" " + x, add_special_tokens=False)[-1] for x in letters]
    ids2 = [tok.encode(x, add_special_tokens=False)[-1] for x in letters]
    prompts = []
    for it in items:
        q = it["question"] + "\n" + "\n".join(f"{l}. {c}" for l, c in zip(letters, it["choices"]))
        prompts.append(chat(tok, q + "\n\nAnswer with the letter only.") + "Answer:")
    correct = 0
    for i in range(0, len(prompts), batch):
        enc = tok(prompts[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
        lg = model(**enc).logits[:, -1].float()
        sc = torch.maximum(lg[:, ids], lg[:, ids2])
        correct += sum(int(p == it["answer"]) for p, it in zip(sc.argmax(-1).tolist(), items[i:i + batch]))
    return 100 * correct / len(items)


def chat_prompts(n=60, seed=0):
    from datasets import load_dataset
    ds = load_dataset("HuggingFaceH4/no_robots", split="test").shuffle(seed=seed)
    out = []
    for r in ds:
        if r["category"] in ("Generation", "Open QA", "Brainstorm") and len(r["prompt"]) < 600:
            out.append(r["prompt"])
        if len(out) == n:
            break
    return out


@torch.no_grad()
def kl_edit(model, tok, seqs, prompt_lens, edit, batch=4):
    """Mean over sequences of mean per-token KL(base || edited) on the response tokens of seqs (teacher-forced)."""
    tok.padding_side = "right"
    vals = []
    try:
        for i in range(0, len(seqs), batch):
            enc = tok(seqs[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(DEV)
            a = torch.log_softmax(model(**enc).logits.float(), -1)
            orig = apply_edit(model, edit)
            b = torch.log_softmax(model(**enc).logits.float(), -1)
            apply_edit(model, orig)
            for j, n in enumerate(enc.attention_mask.sum(1).tolist()):
                p0 = prompt_lens[i + j]
                x, y = a[j, p0 - 1:n - 1], b[j, p0 - 1:n - 1]
                vals.append((x.exp() * (x - y)).sum(-1).mean().item())
            del a, b
    finally:
        tok.padding_side = "left"
    return sum(vals) / len(vals)


def write_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def read_jsonl(path):
    return [json.loads(x) for x in open(path, encoding="utf-8")]


def norm_text(s):
    return re.sub(r"\s+", " ", s.lower()).strip()
