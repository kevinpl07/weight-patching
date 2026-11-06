"""Shared pieces for the less-AI-voice experiments (voice/). Model loading on CPU, chat formatting, batched sampled
generation inside the GPU lock, edits ({param name: full new tensor}), jsonl helpers."""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
from gpu_lock import gpu  # noqa: E402,F401

RUNS = ROOT / "runs" / "voice"
MODEL = os.environ.get("VOICE_MODEL", "google/gemma-3-1b-it")
MEM_FRAC = 0.85


def tag(model):
    return str(model).rstrip("/\\").replace("\\", "/").split("/")[-1]


def load(model=MODEL):
    """(tok, model) with the model in bf16 on the CPU, eval mode."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.bfloat16).eval()
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        m = AutoModelForImageTextToText.from_pretrained(model, dtype=torch.bfloat16).eval()
    return tok, m


def chat(tok, user, system=None):
    msgs = ([dict(role="system", content=system)] if system else []) + [dict(role="user", content=user)]
    kw = {"enable_thinking": False} if "qwen3" in tok.name_or_path.lower() else {}
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)


def stop_ids(model, tok):
    e = model.generation_config.eos_token_id
    e = set(e if isinstance(e, (list, tuple)) else [e]) if e is not None else set()
    return e | {tok.eos_token_id, tok.pad_token_id}


def sampling(model):
    gc = model.generation_config
    return dict(do_sample=bool(gc.do_sample), temperature=gc.temperature, top_p=gc.top_p, top_k=gc.top_k)


@torch.no_grad()
def generate(model, tok, prompts, max_new, seed, label, greedy=False):
    """One batch, inside the GPU lock. Returns [(text, n_tok, truncated, ids)]."""
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
    kw = dict(do_sample=False) if greedy else sampling(model)
    stop = stop_ids(model, tok)
    with gpu(label):
        torch.cuda.set_per_process_memory_fraction(MEM_FRAC)
        model.cuda()
        torch.manual_seed(seed)
        g = model.generate(**{k: v.cuda() for k, v in enc.items()}, max_new_tokens=max_new,
                           pad_token_id=tok.pad_token_id, **kw)
        g = g[:, enc.input_ids.shape[1]:].cpu()
        model.cpu()
        torch.cuda.empty_cache()
    out = []
    for ids in g.tolist():
        n = next((j for j, x in enumerate(ids) if x in stop), len(ids))
        out.append((tok.decode(ids[:n], skip_special_tokens=True).strip(), n, n >= max_new, ids[:n]))
    return out


# ---------------------------------------------------------------- edits

def load_edit(path):
    if path is None:
        return {}
    from safetensors.torch import load_file
    return load_file(str(path))


@torch.no_grad()
def apply_edit(model, edit):
    """Copy {name: full tensor} into the model in place; returns originals for undo."""
    if isinstance(edit, (str, Path)):
        edit = load_edit(edit)
    sd = model.state_dict(keep_vars=True)
    orig = {}
    for k, v in edit.items():
        orig[k] = sd[k].detach().cpu().clone()
        sd[k].data.copy_(v.to(sd[k].dtype))
    return orig


def count_changed(edit, model):
    sd = model.state_dict()
    return sum(int((v.to(sd[k].dtype) != sd[k].cpu()).sum()) for k, v in edit.items())


# ---------------------------------------------------------------- io

def read_jsonl(path):
    path = Path(path)
    return [json.loads(line) for line in path.open(encoding="utf-8")] if path.exists() else []


def append_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def run_gen(model, tok, items, path, max_new=1024, batch=16, seed=0, label="voice", system=None, prompt_key="prompt"):
    """Resumable: generate for items not yet in `path` (by id); one GPU-lock batch at a time."""
    done = {r["id"] for r in read_jsonl(path)}
    todo = [it for it in items if it["id"] not in done]
    for i in range(0, len(todo), batch):
        part = todo[i:i + batch]
        t = time.time()
        res = generate(model, tok, [chat(tok, it[prompt_key], system) for it in part], max_new, seed + i,
                       f"{label} {i}/{len(todo)}")
        append_jsonl(path, [dict(id=it["id"], text=r[0], n_tok=r[1], truncated=r[2]) for it, r in zip(part, res)])
        print(f"  {label}: {i + len(part)}/{len(todo)} ({time.time() - t:.0f}s)", flush=True)
