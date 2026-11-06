"""Model check for the less-AI-voice experiment (PLAN_2026-11-05_less_ai_voice.md, section 3).

Generates answers to 60 single-turn no_robots *test* prompts (Open QA / Brainstorm / Generation / Chat) with each
candidate's shipped sampling settings (generation_config.json: what a user gets by default), and scores AI-voice tells
with voice/tells.py. The human-written no_robots answers to the same prompts are the reference.

    python voice/check.py gen --model meta-llama/Llama-3.2-3B-Instruct
    python voice/check.py summary           # table over every model in runs/voice/check/ + the human reference
Kill rule (plan): every candidate below 1.5 tells per response -> the voice is too weak at this size.
"""
import argparse
import glob
import json
import os
import random
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import torch  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))
from gpu_lock import gpu  # noqa: E402
from tells import score  # noqa: E402

OUT = ROOT / "runs" / "voice"
PROMPTS = OUT / "data" / "check_prompts.json"
QUOTA = {"Open QA": 20, "Brainstorm": 15, "Generation": 15, "Chat": 10}
MEM_FRAC = 0.85


def build_prompts():
    if PROMPTS.exists():
        return json.loads(PROMPTS.read_text(encoding="utf-8"))
    import pandas as pd
    f = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--HuggingFaceH4--no_robots/snapshots/*/data/test-*.parquet"))[0]
    d = pd.read_parquet(f)
    rng = random.Random(0)
    rows, short = [], 0
    for cat, k in QUOTA.items():
        pool = [r for r in d[d.category == cat].itertuples()
                if len(r.messages) == 2 and r.messages[0]["role"] == "user" and len(r.messages[0]["content"]) < 800]
        rng.shuffle(pool)
        take = pool[:k + short]
        short = k + short - len(take)
        rows += [dict(id=r.prompt_id, category=cat, prompt=r.messages[0]["content"], human=r.messages[1]["content"])
                 for r in take]
    if short:  # top up from Generation if a category ran dry
        used = {r["id"] for r in rows}
        pool = [r for r in d[d.category == "Generation"].itertuples()
                if len(r.messages) == 2 and r.prompt_id not in used and len(r.messages[0]["content"]) < 800]
        rows += [dict(id=r.prompt_id, category="Generation", prompt=r.messages[0]["content"],
                      human=r.messages[1]["content"]) for r in pool[:short]]
    PROMPTS.parent.mkdir(parents=True, exist_ok=True)
    PROMPTS.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    return rows


def load(model_id):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16).eval()
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText
        model = AutoModelForImageTextToText.from_pretrained(model_id, dtype=torch.bfloat16).eval()
    return tok, model


def chat(tok, user):
    kw = {}
    if "qwen3" in tok.name_or_path.lower():
        kw["enable_thinking"] = False  # plan: Qwen3 in non-thinking mode
    return tok.apply_chat_template([dict(role="user", content=user)], tokenize=False, add_generation_prompt=True, **kw)


def eos_ids(model, tok):
    e = model.generation_config.eos_token_id
    e = set(e if isinstance(e, (list, tuple)) else [e]) if e is not None else set()
    return e | {tok.eos_token_id, tok.pad_token_id}


@torch.no_grad()
def gen(a):
    rows = build_prompts()
    tag = a.model.rstrip("/").split("/")[-1]
    out = OUT / "check" / tag
    out.mkdir(parents=True, exist_ok=True)
    path = out / "gen.jsonl"
    done = {json.loads(line)["id"] for line in path.open(encoding="utf-8")} if path.exists() else set()
    todo = [r for r in rows if r["id"] not in done]
    if not todo:
        print("nothing to do", tag)
        return
    tok, model = load(a.model)
    gc = model.generation_config
    samp = dict(do_sample=bool(gc.do_sample), temperature=gc.temperature, top_p=gc.top_p, top_k=gc.top_k)
    n_par = sum(p.numel() for p in model.parameters()) / 1e9
    bs = a.batch or (12 if n_par < 2.5 else 6)
    stop = eos_ids(model, tok)
    print(tag, f"{n_par:.2f}B params, batch {bs}, sampling {samp}", flush=True)
    for i in range(0, len(todo), bs):
        chunk = todo[i:i + bs]
        enc = tok([chat(tok, r["prompt"]) for r in chunk], return_tensors="pt", padding=True, add_special_tokens=False)
        with gpu(f"voice check {tag} {i}"):
            torch.cuda.set_per_process_memory_fraction(MEM_FRAC)
            t = time.time()
            model.cuda()
            torch.manual_seed(a.seed + i)
            g = model.generate(**{k: v.cuda() for k, v in enc.items()}, max_new_tokens=a.max_new,
                               pad_token_id=tok.pad_token_id, **samp)
            g = g[:, enc.input_ids.shape[1]:].cpu()
            model.cpu()
            torch.cuda.empty_cache()
            secs = time.time() - t
        recs = []
        for r, ids in zip(chunk, g.tolist()):
            n = next((j for j, x in enumerate(ids) if x in stop), len(ids))
            text = tok.decode(ids[:n], skip_special_tokens=True).strip()
            recs.append(dict(id=r["id"], category=r["category"], text=text, n_tok=n, truncated=n >= a.max_new,
                             sampling=samp))
        with path.open("a", encoding="utf-8") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {i + len(chunk)}/{len(todo)}  {secs:.0f}s", flush=True)


FAMS = ("opener", "closer", "lexicon", "construction", "register", "emdash", "emoji", "markdown_unasked")


def summarize_rows(texts, prompts):
    S = [score(t, p) for t, p in zip(texts, prompts)]
    n = len(S)
    words = sum(s["words"] for s in S)
    row = dict(n=n, tells=sum(s["total"] for s in S) / n, words=words / n)
    for f in FAMS:
        row[f] = sum(s[f] for s in S) / n
    row["opener_%"] = 100 * sum(s["opener"] > 0 for s in S) / n
    row["closer_%"] = 100 * sum(s["closer"] > 0 for s in S) / n
    row["md_unasked_%"] = 100 * sum(s["markdown_unasked"] for s in S) / max(1, sum(not s["structure_asked"] for s in S))
    row["lex_per_1k"] = 1000 * sum(s["lexicon"] for s in S) / max(1, words)
    row["emdash_per_1k"] = 1000 * sum(s["emdash"] for s in S) / max(1, words)
    row["tells_per_100w"] = 100 * sum(s["total"] for s in S) / max(1, words)
    return row, S


def summary(a):
    rows = build_prompts()
    by_id = {r["id"]: r for r in rows}
    table = {"human (no_robots)": summarize_rows([r["human"] for r in rows], [r["prompt"] for r in rows])[0]}
    trunc = {}
    for p in sorted((OUT / "check").glob("*/gen.jsonl")):
        recs = [json.loads(line) for line in p.open(encoding="utf-8")]
        table[p.parent.name], S = summarize_rows([r["text"] for r in recs], [by_id[r["id"]]["prompt"] for r in recs])
        trunc[p.parent.name] = 100 * sum(r["truncated"] for r in recs) / len(recs)
        # pattern breakdown: top hits
        from collections import Counter
        c = Counter(h for s in S for h in s["hits"])
        table[p.parent.name]["top_hits"] = ", ".join(f"{k} {v}" for k, v in c.most_common(8))
    cols = ["n", "tells", "tells_per_100w", "words", "opener_%", "closer_%", "md_unasked_%", "lex_per_1k",
            "emdash_per_1k", "construction", "register", "emoji"]
    lines = ["| model | " + " | ".join(cols) + " | trunc % |", "|" + "---|" * (len(cols) + 2)]
    for k, r in table.items():
        lines.append(f"| {k} | " + " | ".join(f"{r[c]:.2f}" if isinstance(r[c], float) else str(r[c]) for c in cols)
                     + f" | {trunc.get(k, 0):.0f} |")
    lines.append("")
    for k, r in table.items():
        if "top_hits" in r:
            lines.append(f"- **{k}** top pattern hits: {r['top_hits']}")
    txt = "\n".join(lines)
    print(txt)
    (OUT / "check").mkdir(parents=True, exist_ok=True)
    (OUT / "check" / "summary.md").write_text(txt + "\n", encoding="utf-8")
    (OUT / "check" / "summary.json").write_text(json.dumps(table, indent=1), encoding="utf-8")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--model", required=True)
    g.add_argument("--max-new", type=int, default=1024)
    g.add_argument("--batch", type=int, default=0)
    g.add_argument("--seed", type=int, default=0)
    sub.add_parser("summary")
    sub.add_parser("prompts")
    a = ap.parse_args()
    if a.cmd == "gen":
        gen(a)
    elif a.cmd == "summary":
        summary(a)
    else:
        rows = build_prompts()
        print(len(rows), {c: sum(r["category"] == c for r in rows) for c in QUOTA})
