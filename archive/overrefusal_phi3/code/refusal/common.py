"""Shared helpers for the over-refusal experiment (PLAN_2026-11-04_practical_edits.md, section 1).

Models: meta-llama/Llama-3.2-1B-Instruct (target), google/gemma-3-1b-it (backup, first-class), Qwen/Qwen3-0.6B (dev).
All three tie lm_head to embed_tokens.

TIED EMBEDDINGS (one rule everywhere): when lm_head.weight and model.embed_tokens.weight are the same tensor, the only
name an edit may use is `model.embed_tokens.weight`. `lm_head.weight` is an alias; edit files that contain it are
rejected. Changing row t of the embedding changes both what token t writes into the residual stream at the input and
the logit of token t at the output, and its entries count once against any budget. (Gemma multiplies the input
embedding by sqrt(hidden_size); lm_head uses the raw rows.)

Chat format: every prompt goes through the model's own chat template with add_generation_prompt=True,
enable_thinking=False (Qwen3) and a fixed date_string (Llama 3.2's template otherwise inserts today's date). Gemma 3
has no system role; its template folds a system message into the first user turn ("<system>\n\n<user>"), which is how
the system-prompt arm reaches Gemma. Template strings already contain BOS, so they are tokenized with
add_special_tokens=False.
"""
import contextlib
import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gpu_lock import gpu  # noqa: E402

RUNS = ROOT / "runs" / "refusal"
DATA = RUNS / "data"
DEFAULT_MODEL = os.environ.get("REFUSAL_MODEL", "meta-llama/Llama-3.2-1B-Instruct")
MEM_FRAC = 0.85
DATE = "26 Jul 2024"


def slug(model):
    return model.rstrip("/").split("/")[-1]


def jdump(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")


def jload(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- model loading

def load_tok(model_name):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    tok.padding_side = "left"
    if tok.pad_token is None:
        cand = "<|finetune_right_pad_id|>"  # Llama 3.x reserved pad token
        tok.pad_token = cand if cand in tok.get_vocab() else tok.eos_token
    return tok


def load_model(model_name, edit=None):
    """CPU bf16 model in eval mode, optionally with an edit ({name: full tensor} or a path) applied."""
    from transformers import AutoModelForCausalLM
    from transformers import AutoConfig
    kw = {}
    if AutoConfig.from_pretrained(model_name).model_type == "gemma2":
        kw["attn_implementation"] = "eager"  # Gemma 2's attention logit softcapping is only implemented in eager
    m = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, **kw).eval()
    m.requires_grad_(False)
    if edit is not None:
        apply_edit(m, edit)
    return m


def load(model_name, edit=None):
    return load_tok(model_name), load_model(model_name, edit)


def is_tied(model):
    out, inp = model.get_output_embeddings(), model.get_input_embeddings()
    if out is None:
        return False
    if out.weight is inp.weight:
        return True
    return out.weight.device.type != "meta" and out.weight.data_ptr() == inp.weight.data_ptr()


def params(model):
    """{name: parameter} with each tensor once (lm_head.weight dropped when tied)."""
    tied = is_tied(model)
    return {k: v for k, v in model.named_parameters(remove_duplicate=False)
            if not (tied and k == "lm_head.weight")}


def n_params(model):
    return sum(v.numel() for v in params(model).values())


def load_edit(path):
    if path is None:
        return {}
    if isinstance(path, dict):
        return path
    from safetensors.torch import load_file
    return load_file(str(path))


def save_edit_file(edit, path):
    from safetensors.torch import save_file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().to("cpu", torch.bfloat16).contiguous() for k, v in edit.items()}, str(path))


def check_edit(model, edit):
    """Validate names/shapes/finite values under the tied-embedding rule. Returns {name: n entries changed}."""
    P = params(model)
    tied = is_tied(model)
    changed = {}
    for k, v in edit.items():
        if tied and k == "lm_head.weight":
            raise ValueError("lm_head.weight is tied to model.embed_tokens.weight in this model; edit "
                             "model.embed_tokens.weight instead (it changes both the input embedding and the output logits)")
        if k not in P:
            raise ValueError(f"unknown parameter {k}")
        if tuple(v.shape) != tuple(P[k].shape):
            raise ValueError(f"shape mismatch for {k}: {tuple(v.shape)} vs {tuple(P[k].shape)}")
        if not torch.isfinite(v.float()).all():
            raise ValueError(f"non-finite values in {k}")
        changed[k] = int((v.to(torch.bfloat16) != P[k].data.cpu()).sum())
    return changed


@torch.no_grad()
def apply_edit(model, edit):
    """Apply an edit file (path or {name: full new tensor}) in place. Used by eval.py, baselines.py, agent_tools.py.
    Returns {name: entries changed}."""
    edit = load_edit(edit)
    changed = check_edit(model, edit)
    P = params(model)
    for k, v in edit.items():
        P[k].data.copy_(v.to(P[k].dtype))
    return changed


def checkpoint_map(model):
    """{param name: safetensors shard path} for the model's own cached checkpoint, or None if a parameter is missing
    (then originals are copied to RAM instead). Lets base weights be restored from disk without a RAM copy."""
    try:
        import json as _json
        from huggingface_hub import snapshot_download
        d = Path(snapshot_download(model.config._name_or_path, local_files_only=True,
                                   allow_patterns=["*.safetensors", "*.json"]))
        idx = d / "model.safetensors.index.json"
        if idx.exists():
            wm = _json.loads(idx.read_text())["weight_map"]
            m = {k: str(d / f) for k, f in wm.items()}
        else:
            from safetensors import safe_open
            with safe_open(str(d / "model.safetensors"), "pt") as f:
                m = {k: str(d / "model.safetensors") for k in f.keys()}
        return m if all(k in m for k in params(model)) else None
    except Exception:
        return None


def original_tensor(cmap, key):
    from safetensors import safe_open
    with safe_open(cmap[key], "pt") as f:
        return f.get_tensor(key)


@torch.no_grad()
def restore(model, keys, cmap=None):
    """Reset the given parameters to the original checkpoint values (read from disk)."""
    cmap = cmap or checkpoint_map(model)
    P = params(model)
    for k in keys:
        P[k].data.copy_(original_tensor(cmap, k).to(P[k].device, P[k].dtype))


class Swap:
    """One model that can switch between base weights and an edit in place, so base-vs-edited comparisons never
    need a second copy of a 3-4B model. Originals come from the cached checkpoint on disk when possible (else a RAM
    copy of the edited tensors), and path edits are streamed from disk, so a multi-GB edit costs no extra RAM.
    Works on CPU or GPU."""

    def __init__(self, model, edit=None):
        self.model = model
        self.P = params(model)
        self.edit, self.orig, self.state, self.path, self.keys = {}, {}, "base", None, []
        self.cmap = checkpoint_map(model)
        if edit is not None:
            self.set_edit(edit)

    @torch.no_grad()
    def set_edit(self, edit):
        """edit: {name: tensor} (kept in RAM) or a .safetensors path (streamed tensor by tensor on every switch, so a
        multi-GB abliteration edit of a 3.8B model is never held in RAM next to the model and its originals)."""
        self.use("base")
        self.path = None
        if isinstance(edit, (str, Path)):
            from safetensors import safe_open
            self.path, self.changed = str(edit), {}
            with safe_open(self.path, "pt") as f:
                for k in f.keys():
                    self.changed.update(check_edit(self.model, {k: f.get_tensor(k)}))
            keys = list(self.changed)
            self.edit = {}
        else:
            self.changed = check_edit(self.model, edit)
            self.edit = {k: v.to(torch.bfloat16) for k, v in edit.items()}
            keys = list(self.edit)
        self.keys = keys
        self.orig = {} if self.cmap else {k: self.P[k].data.to("cpu", copy=True) for k in keys}
        return self.changed

    def _orig_tensors(self):
        if self.cmap:
            for k in self.keys:
                yield k, original_tensor(self.cmap, k)
        else:
            yield from self.orig.items()

    def _edited_tensors(self):
        if self.path is None:
            yield from self.edit.items()
        else:
            from safetensors import safe_open
            with safe_open(self.path, "pt") as f:
                for k in f.keys():
                    yield k, f.get_tensor(k)

    @torch.no_grad()
    def use(self, which):
        if which == self.state:
            return self.model
        src = self._edited_tensors() if which == "edited" else self._orig_tensors()
        for k, v in src:
            self.P[k].data.copy_(v.to(self.P[k].device, self.P[k].dtype))
        self.state = which
        return self.model


@contextlib.contextmanager
def on_gpu(models, label):
    """Take the GPU lock, move model(s) to CUDA, yield, move back to CPU and free memory."""
    models = models if isinstance(models, (list, tuple)) else [models]
    with gpu(label):
        torch.cuda.set_per_process_memory_fraction(MEM_FRAC)
        t = time.time()
        try:
            for m in models:
                m.cuda()
            yield
        finally:
            for m in models:
                m.cpu()
            torch.cuda.empty_cache()
            on_gpu.seconds += time.time() - t


on_gpu.seconds = 0.0


# ---------------------------------------------------------------- prompting and generation

def _template(tok, msgs):
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False,
                                   date_string=DATE)


def chat(tok, user, system=None):
    """Chat-formatted prompt. If the template has no system role (it raises, as Gemma 2's does, or silently drops the
    system text), the system text is folded into the user turn as "<system>\n\n<user>", which is what Gemma 3's own
    template does."""
    if not system:
        return _template(tok, [{"role": "user", "content": user}])
    try:
        out = _template(tok, [{"role": "system", "content": system}, {"role": "user", "content": user}])
        if system.strip()[:40] in out:
            return out
    except Exception:
        pass
    return _template(tok, [{"role": "user", "content": system + "\n\n" + user}])


def auto_batch(model, seq_len, cap=64):
    """Largest batch whose KV cache fits next to the weights in MEM_FRAC of the card (bf16, all layers full-length)."""
    cfg = model.config
    hd = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    kv = 2 * cfg.num_hidden_layers * getattr(cfg, "num_key_value_heads", cfg.num_attention_heads) * hd * 2
    total = torch.cuda.get_device_properties(0).total_memory * MEM_FRAC
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    free = total - weights - 1.5e9
    return max(1, min(cap, int(free / (kv * seq_len * 1.3))))


def eos_ids(model, tok):
    e = model.generation_config.eos_token_id
    e = list(e) if isinstance(e, (list, tuple)) else [e]
    if tok.eos_token_id is not None:
        e.append(tok.eos_token_id)
    vocab = tok.get_vocab()
    for t in ("<|end|>", "<|eot_id|>", "<end_of_turn>", "<|im_end|>"):  # end-of-turn markers (Phi-3, Llama 3, Gemma, Qwen)
        if t in vocab:
            e.append(vocab[t])
    return sorted({x for x in e if x is not None})


def encode(tok, texts, device="cuda"):
    return tok(list(texts), return_tensors="pt", padding=True, padding_side="left",
               add_special_tokens=False).to(device)


@torch.no_grad()
def generate(model, tok, prompts, max_new=256, batch=64, temperature=0.0, seed=0, verbose=True):
    """Greedy (or sampled) generation for chat-formatted prompt strings. Model must already be on CUDA.
    Returns [(text, n_tokens, truncated, ids)] in input order; ids include the EOS token if one was produced."""
    eos = eos_ids(model, tok)
    order = sorted(range(len(prompts)), key=lambda i: -len(prompts[i]))  # longest first: OOM early, not late
    res = [None] * len(prompts)
    if temperature == 0:
        samp = dict(do_sample=False, temperature=None, top_p=None, top_k=None)
    else:
        samp = dict(do_sample=True, temperature=temperature, top_p=1.0, top_k=0)
    s = 0
    while s < len(order):
        L = len(tok(prompts[order[s]], add_special_tokens=False).input_ids)
        b = min(batch, auto_batch(model, L + max_new))
        idx = order[s:s + b]
        torch.manual_seed(seed + s)
        enc = encode(tok, [prompts[i] for i in idx])
        out = model.generate(**enc, max_new_tokens=max_new, eos_token_id=eos, pad_token_id=tok.pad_token_id,
                             use_cache=True, **samp)
        for j, c in enumerate(out[:, enc.input_ids.shape[1]:].tolist()):
            cut = next((p for p, t in enumerate(c) if t in eos), None)
            ids = c[:cut] if cut is not None else c
            res[idx[j]] = (tok.decode(ids, skip_special_tokens=True), len(ids), cut is None,
                           c[:cut + 1] if cut is not None else c)
        s += len(idx)
        if verbose:
            print(f"    gen {s}/{len(order)} (batch {len(idx)})", flush=True)
    return res


# ---------------------------------------------------------------- residual stream structure

def layers(model):
    return model.model.layers


def norm_gain(norm_module):
    """Elementwise gain of an RMSNorm: Gemma stores w and multiplies by (1 + w); Llama/Qwen multiply by w."""
    w = norm_module.weight.detach().float()
    return 1.0 + w if "Gemma" in type(norm_module).__name__ else w


def residual_writers(model):
    """Every matrix whose output is added to the residual stream, as (param_name, out_axis, post_norm_module|None).
    out_axis: the axis of the hidden dimension (0 for nn.Linear weights [hidden, in]; 1 for embedding rows).
    Gemma 2/3 put an RMSNorm *after* o_proj and down_proj (post_attention_layernorm / post_feedforward_layernorm,
    detected by the presence of pre_feedforward_layernorm). In Llama/Qwen, post_attention_layernorm is the pre-MLP
    norm and the writers add straight into the residual stream."""
    out = [("model.embed_tokens.weight", 1, None)]
    for i, L in enumerate(layers(model)):
        sandwich = hasattr(L, "pre_feedforward_layernorm")
        out.append((f"model.layers.{i}.self_attn.o_proj.weight", 0, L.post_attention_layernorm if sandwich else None))
        out.append((f"model.layers.{i}.mlp.down_proj.weight", 0, L.post_feedforward_layernorm if sandwich else None))
    return out


def orthogonalize(W, u, axis, along=None, alpha=1.0):
    """Remove the component along unit vector u from the hidden axis of W (float32 math).
    along=None: orthogonal projection W - u u^T W (standard abliteration).
    along=v: oblique projection W - v (u^T W) / (v . u): the output o' satisfies o'.u = 0 exactly, and o changes only
    along v (used for writers followed by a post-norm, see writer_projection).
    alpha < 1: partial strength W - alpha * (...) (dose curves); alpha = 1 is the full projection."""
    W = W.float()
    u = u.float() / u.float().norm()
    v = u if along is None else along.float() / (along.float() @ u)
    if axis == 0:
        return W - alpha * torch.outer(v, u @ W)
    return W - alpha * torch.outer(W @ u, v)


POSTNORM_REG = 0.1


def writer_projection(r, post_norm):
    """(read, along) vectors for orthogonalizing a residual writer against direction r.
    Without a post-norm: read = r, along = r (orthogonal projection, standard abliteration).
    With an RMSNorm after the writer (Gemma 2/3), the residual receives w = g * o / rms(o) (g = elementwise gain,
    1 + weight in Gemma), and w . r = o . (g*r) / rms(o). So the writer's output must be orthogonal to read = g*r,
    which removes r from what reaches the residual exactly (the RMS rescale is a scalar per token). Which way o moves
    is a free choice: along g*r (orthogonal projection) changes w by g*g*r-shaped amounts, large because Gemma's gains
    range from ~0 to ~200; along r/g would change w only along r (the ideal) but blows up where g ~ 0. We use the
    regularized compromise along = r*g / (g^2 + 0.1 median(g^2)), fixed in advance. On random inputs and directions
    across all 52 Gemma-3-1B writers, the change of w beyond the ideal (r removed, nothing else) is a median 0.29x
    of the ideal change (orthogonal g*r: 1.18x, max 24x). The RMS rescale means the other components of w still get
    scaled up slightly."""
    r = r.float() / r.float().norm()
    if post_norm is None:
        return r, r
    g = norm_gain(post_norm)
    along = r * g / (g * g + POSTNORM_REG * (g * g).median())
    return g * r, along


@torch.no_grad()
def last_token_resid(model, tok, prompts, batch=32, system=None):
    """Residual stream at the last prompt token (the end of the generation prompt), for every layer boundary:
    returns float32 [N, n_layers + 1, hidden], index l = input to layer l, index n_layers = output of the last layer
    (before the final norm). Model must be on CUDA."""
    Ls = layers(model)
    caps = {}

    def pre(i):
        def f(mod, args, kwargs):
            h = args[0] if args else kwargs["hidden_states"]
            caps[i] = h[:, -1].float().cpu()
        return f

    def post(mod, args, kwargs, out):
        h = out[0] if isinstance(out, tuple) else out
        caps[len(Ls)] = h[:, -1].float().cpu()

    hooks = [L.register_forward_pre_hook(pre(i), with_kwargs=True) for i, L in enumerate(Ls)]
    hooks.append(Ls[-1].register_forward_hook(post, with_kwargs=True))
    res = []
    try:
        for s in range(0, len(prompts), batch):
            enc = encode(tok, [chat(tok, p, system) for p in prompts[s:s + batch]])
            model(**enc, logits_to_keep=1, use_cache=False)
            res.append(torch.stack([caps[i] for i in range(len(Ls) + 1)], 1))
    finally:
        for h in hooks:
            h.remove()
    return torch.cat(res)
