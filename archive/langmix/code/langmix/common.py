"""Shared pieces for the language-mixing experiment (experiment 2 of PLAN_2026-11-04_practical_edits.md).

Settings come from langmix/check.py (imported, never modified) so that every arm uses exactly the viability check's
prompt suffix, sampling, CJK regexes and answer extraction, and the check's 200 MATH-500 rows double as the base arm.

What lives here:
- prompt building (English / Chinese suffix, optional instruction in the user turn or system role)
- model loading on CPU, `apply_edit` (the one edit-application helper used by eval, session, agent tools, baselines)
- `cjk_token_mask`: every vocab token whose bytes start (part of) a CJK character, for the token-ban arm
- `Steer`: a forward hook that adds a vector to the residual stream at one layer, per row (for the steering arm)
- `generate_chunk`: one batch of generations inside the GPU lock (model moved to GPU and back)
"""
import json
import os

import sys
import time
from pathlib import Path

# check.py hit the 0.85 cap at batch 50 x 2,048 (SDPA repeat_kv expands the KV cache 6x); the orchestrator reran it
# with batch 24 and expandable segments. Entry scripts also set this before importing torch.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from gpu_lock import gpu  # noqa: E402
from check import CJK, HAN_RUN, SAMPLING, SUFFIX, last_boxed  # noqa: E402,F401  (check.py's exact settings)

MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
RUNS = ROOT / "runs" / "langmix"
DATA = RUNS / "data"
MEM_FRAC = 0.85
MAX_NEW = 2048
BATCH = 24          # max safe batch for 2,048-token generation on the 12 GB card (check.py OOMed at 50)
SEED0 = 1000        # seed of a chunk = 1000 + index of its first item (check.py's convention for a fresh run)
# Chinese counterpart of check.SUFFIX, used for every Chinese prompt (MGSM-zh, Chinese practice prompts).
ZH_SUFFIX = "\n请逐步推理，并将最终答案放在 \\boxed{} 中。"
# Answer-format suffix per prompt language (a multilingual framing adds its languages here; unknown -> English).
SUFFIXES = dict(en=SUFFIX, zh=ZH_SUFFIX)

INSTRUCTIONS = {
    # the instruction arm of the de/ru/th framing (orchestrator's wording), appended to the user turn
    "same": "Reply in the same language as the question.",
    # first framing (English MATH) only
    "en": "Answer only in English.",
}


def model_tag(model):
    """Short name of a hub id or a local checkpoint dir (e.g. runs/langmix/models/tina-rs1-2400 -> tina-rs1-2400),
    the same convention as check.py's a.model.split("/")[-1]."""
    return Path(str(model).rstrip("/").rstrip(chr(92))).name


# ---------------------------------------------------------------- prompts

def user_text(item):
    """The user turn for a suite item (before any arm instruction)."""
    kind = item.get("kind", "math")
    if kind == "math":
        return item["question"] + SUFFIXES.get(item.get("lang", "en"), SUFFIX)
    return item["question"]  # ifeval / chat: the prompt as is


def chat_prompt(tok, user, instruction=None, role="user"):
    """Chat-formatted prompt string. DeepSeek advises against system prompts for R1 models ("all instructions should
    be contained within the user prompt"), so by default an arm instruction is appended to the user turn on its own
    line; role="system" puts it in the system slot instead."""
    msgs = []
    if instruction and role == "system":
        msgs.append(dict(role="system", content=instruction))
    elif instruction:
        user = user + "\n" + instruction
    msgs.append(dict(role="user", content=user))
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# ---------------------------------------------------------------- model + edits

def load_tok(model=MODEL):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_model(model=MODEL):
    """bf16 model on CPU, eval mode. Move it to the GPU only inside `gpu(...)`."""
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).eval()


def load_edit(path):
    """{param name: full new tensor} from .safetensors (agent/random/scripted edits) or .pt."""
    if path is None:
        return {}
    path = Path(path)
    if path.suffix == ".safetensors":
        from safetensors.torch import load_file
        return load_file(str(path))
    return torch.load(path, map_location="cpu")


@torch.no_grad()
def apply_edit(model, edit):
    """Copy an edit ({name: full new tensor}, or a path) into the model's parameters in place.
    Returns {name: original tensor (CPU clone)} so the caller can undo it with apply_edit(model, originals).
    lm_head is untied from embed_tokens in R1-Distill-Qwen-1.5B (tie_word_embeddings=false), so an lm_head edit
    does not touch the input embeddings. In a tied model (the backup, Qwen3-0.6B) lm_head.weight and
    model.embed_tokens.weight are one tensor; an edit may name either, and both names change together."""
    if isinstance(edit, (str, Path)):
        edit = load_edit(edit)
    sd = model.state_dict(keep_vars=True)
    if is_tied(model) and "lm_head.weight" in edit and "model.embed_tokens.weight" in edit:
        raise ValueError("tied model: give lm_head.weight or model.embed_tokens.weight, not both")
    orig = {}
    for k, v in edit.items():
        if k not in sd:
            raise KeyError(f"unknown parameter {k}")
        p = sd[k]
        if tuple(v.shape) != tuple(p.shape):
            raise ValueError(f"shape mismatch for {k}: {tuple(v.shape)} vs {tuple(p.shape)}")
        orig[k] = p.detach().to("cpu").clone()
        p.data.copy_(v.to(p.device, p.dtype))
    return orig


def is_tied(model):
    return model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()


def count_changed(edit, base):
    return {k: int((v.to(base[k].dtype) != base[k]).sum()) for k, v in edit.items()}


# ---------------------------------------------------------------- CJK token ban

def _byte_decoder():
    """Inverse of GPT-2's bytes_to_unicode (Qwen2 uses byte-level BPE)."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b); cs.append(256 + n); n += 1
    return {chr(c): b for b, c in zip(bs, cs)}


def cjk_token_ids(tok):
    """Every vocab token that can emit a CJK character (check.CJK: kana, Han incl. ext. A, Hangul):
    tokens whose decoded text contains one, plus byte-fragment tokens containing a UTF-8 lead byte 0xE3-0xED
    (the lead bytes of every code point U+3000-U+D7FF, i.e. CJK punctuation, kana, Han, Hangul), so that a
    Han character spelled with byte tokens cannot slip through either. Special tokens are never banned.
    Cached in runs/langmix/data/cjk_ids_<tokenizer>.json."""
    cache = DATA / f"cjk_ids_{model_tag(tok.name_or_path)}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    dec = _byte_decoder()
    special = set(tok.all_special_ids) | set(tok.added_tokens_decoder)
    ids = []
    for t, i in tok.get_vocab().items():
        if i in special:
            continue
        try:
            raw = bytes(dec[c] for c in t)
        except KeyError:
            continue
        text = raw.decode("utf-8", errors="ignore")
        if CJK.search(text) or any(0xE3 <= b <= 0xED for b in raw):
            ids.append(i)
    ids.sort()
    DATA.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(ids))
    return ids


class BanProcessor:
    """LogitsProcessor that sets the logits of the given token ids to -inf (same effect as bad_words_ids with
    single-token entries, but one vectorised mask instead of ~30k word checks per step)."""

    def __init__(self, ids, vocab):
        self.mask = torch.zeros(vocab, dtype=torch.bool)
        self.mask[torch.tensor(ids)] = True

    def __call__(self, input_ids, scores):
        if self.mask.device != scores.device:
            self.mask = self.mask.to(scores.device)
        return scores.masked_fill(self.mask[: scores.shape[-1]], float("-inf"))


# ---------------------------------------------------------------- steering hook

class Steer:
    """Adds alpha_b * v to the output of decoder layer `layer` (the residual stream after that layer), at every
    position (prompt and generated), with a per-row alpha (so a batch can mix English and Chinese prompts).
    Use: with Steer(model, layer, v, alphas): model.generate(...)."""

    def __init__(self, model, layer, vec, alphas):
        self.model, self.layer = model, layer
        self.vec = vec.float()
        self.alphas = torch.as_tensor(alphas, dtype=torch.float32)

    def _hook(self, mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        add = (self.alphas.to(h.device)[:, None, None] * self.vec.to(h.device)[None, None, :]).to(h.dtype)
        h = h + add
        return (h,) + tuple(out[1:]) if isinstance(out, tuple) else h

    def __enter__(self):
        self.h = self.model.model.layers[self.layer].register_forward_hook(self._hook)
        return self

    def __exit__(self, *a):
        self.h.remove()


# ---------------------------------------------------------------- generation

def decode_row(tok, ids, max_new):
    """check.py's convention: cut at the first pad/eos; truncated = no eos within max_new tokens."""
    ids = ids.tolist() if hasattr(ids, "tolist") else list(ids)
    n_tok = next((i for i, x in enumerate(ids) if x in (tok.pad_token_id, tok.eos_token_id)), len(ids))
    return tok.decode(ids[:n_tok], skip_special_tokens=True), n_tok, n_tok >= max_new


@torch.no_grad()
def generate_chunk(model, tok, prompts, seed, max_new=MAX_NEW, label="langmix", processors=None, steer=None):
    """One batch of sampled generations with check.py's settings, inside the GPU lock. The model lives on the CPU
    and is moved to the GPU and back within the lock. steer = (layer, vec, alphas) or None.
    Returns [(text, n_tok, truncated)] in input order, and the seconds spent on the GPU."""
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    kw = dict(max_new_tokens=max_new, pad_token_id=tok.pad_token_id, **SAMPLING)
    if processors:
        from transformers import LogitsProcessorList
        kw["logits_processor"] = LogitsProcessorList(processors)
    with gpu(label):
        torch.cuda.set_per_process_memory_fraction(MEM_FRAC)
        t = time.time()
        model.cuda()
        torch.manual_seed(seed)
        if steer is not None:
            with Steer(model, *steer):
                gen = model.generate(**{k: v.cuda() for k, v in enc.items()}, **kw)
        else:
            gen = model.generate(**{k: v.cuda() for k, v in enc.items()}, **kw)
        gen = gen[:, enc.input_ids.shape[1]:].cpu()
        model.cpu()
        secs = time.time() - t
    return [decode_row(tok, g, max_new) for g in gen], secs


def read_jsonl(path):
    path = Path(path)
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def append_jsonl(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
