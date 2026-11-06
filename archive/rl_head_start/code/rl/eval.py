"""Fixed scoring sets for the RL-debugging experiment. The agent never sees these.

    python rl/eval.py --model Qwen/Qwen3.5-0.8B --name base                  # all suites, greedy + sampled
    python rl/eval.py --model runs/grpo/run1/ckpt/step_00200 --name run1_s200
    python rl/eval.py --model Qwen/Qwen3.5-0.8B --name base --suites thinkprobe

Suites (sets are fixed; do not change them between checkpoints):
- gsm8k:     GSM8K test, all 1,319 problems. Same system prompt and answer format as training, non-thinking, 512 tokens.
- symbolic:  GSM-Symbolic "main", instances 0-4 of each of the 100 templates (500 problems). Same settings.
- ifeval:    all 541 IFEval prompts, no system prompt, non-thinking, 1,024 tokens, greedy only. Strict + loose scores.
- thinkprobe: first 128 GSM8K test problems in thinking mode, 4,096 tokens, temperature 0.6/top-p 0.95/top-k 20
             (Sparks' settings minus presence_penalty, which HF generate lacks). Measures the thinking-loop failure.

Modes for gsm8k/symbolic: greedy, and one sample per problem at temperature 1 with no top-k/top-p (the training
distribution). Symptom metrics: share truncated at the token limit, share truncated in a loop (a 32-token span
repeated 3+ times in the last 256 tokens), mean length, share containing CJK characters (language mixing).

Output: runs/eval/<name>/<suite>_<mode>.jsonl.gz (every answer) and runs/eval/<name>/summary.json.
"""
import argparse
import functools
import gzip
import json
import sys
import time
import types
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from grpo import CJK, MODEL, SYSTEM, score  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SAMPLE = dict(do_sample=True, temperature=1.0, top_k=0, top_p=1.0)
GREEDY = dict(do_sample=False)
THINK = dict(do_sample=True, temperature=0.6, top_k=20, top_p=0.95)


# ---------------------------------------------------------------- data

def gsm8k_test():
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    return [dict(id=f"gsm8k-{i}", question=r["question"], gold=r["answer"].split("####")[-1].strip().replace(",", ""))
            for i, r in enumerate(ds)]


def symbolic():
    from datasets import load_dataset
    ds = load_dataset("apple/GSM-Symbolic", "main", split="test")
    return [dict(id=f"sym-{r['original_id']}-{r['instance']}", question=r["question"],
                 gold=r["answer"].split("####")[-1].strip().replace(",", ""))
            for r in ds if int(r["instance"]) < 5]


def ifeval_docs():
    from datasets import load_dataset
    return [dict(r) for r in load_dataset("google/IFEval", split="train")]


@functools.cache  # the stub modules below break find_spec on a second call
def ifeval_scorer():
    """lm-eval's IFEval checker, imported without lm_eval.tasks.__init__ (which needs the whole harness)."""
    import importlib.util
    pkg = Path(importlib.util.find_spec("lm_eval").submodule_search_locations[0])
    for name, path in [("lm_eval", pkg), ("lm_eval.tasks", pkg / "tasks")]:
        if name not in sys.modules or not hasattr(sys.modules[name], "__path__"):
            m = types.ModuleType(name); m.__path__ = [str(path)]; sys.modules[name] = m
    from lm_eval.tasks.ifeval import utils
    return utils.process_results


# ---------------------------------------------------------------- generation

def is_loop(ids, tail=256, n=32, k=3):
    t = ids[-tail:]
    grams = Counter(tuple(t[i:i + n]) for i in range(0, len(t) - n + 1))
    return bool(grams) and max(grams.values()) >= k


@torch.no_grad()
def generate(model, tok, prompts, max_new, sampling, batch, seed):
    """prompts: list of chat-formatted strings. Returns list of (completion_ids, truncated) in input order."""
    model.eval()  # train mode + gradient checkpointing would silently drop the KV cache
    eos = [tok.convert_tokens_to_ids("<|im_end|>"), tok.pad_token_id]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    res = [None] * len(prompts)
    for s in range(0, len(order), batch):
        idx = order[s:s + batch]
        torch.manual_seed(seed + s)
        enc = tok([prompts[i] for i in idx], return_tensors="pt", padding=True, padding_side="left").to("cuda")
        out = model.generate(**enc, max_new_tokens=max_new, eos_token_id=eos, pad_token_id=tok.pad_token_id,
                             repetition_penalty=1.0, use_cache=True, **sampling)
        for j, c in enumerate(out[:, enc.input_ids.shape[1]:].tolist()):
            cut = next((p for p, t in enumerate(c) if t in eos), None)
            res[idx[j]] = (c[:cut] if cut is not None else c, cut is None)
        print(f"    {min(s + batch, len(order))}/{len(order)}", flush=True)
    return res


def chat(tok, user, system=None, thinking=False):
    msgs = ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)


def symptoms(recs):
    n = len(recs)
    return dict(n=n, trunc=sum(r["truncated"] for r in recs) / n, loop=sum(r["loop"] for r in recs) / n,
                len=sum(r["length"] for r in recs) / n, cjk=sum(r["cjk"] for r in recs) / n)


def run_math(model, tok, probs, mode, args, thinking=False):
    sampling = dict(greedy=GREEDY, sample=dict(SAMPLE, temperature=getattr(args, "temperature", 1.0)), think=THINK)[mode]
    max_new = 4096 if thinking else 512
    batch = min(args.batch, 32) if thinking else args.batch
    gens = generate(model, tok, [chat(tok, p["question"], SYSTEM, thinking) for p in probs], max_new, sampling,
                    batch, args.seed)
    recs = []
    for p, (ids, trunc) in zip(probs, gens):
        text = tok.decode(ids, skip_special_tokens=True)
        answer = text.split("</think>")[-1] if thinking else text
        _, correct, boxed = score(answer, p["gold"], 0.0)
        recs.append(dict(id=p["id"], gold=p["gold"], correct=correct, boxed=boxed, truncated=trunc,
                         loop=trunc and is_loop(ids), length=len(ids), cjk=bool(CJK.search(text)), completion=text))
    summ = dict(acc=sum(r["correct"] for r in recs) / len(recs), boxed=sum(r["boxed"] for r in recs) / len(recs),
                **symptoms(recs))
    return recs, summ


def run_ifeval(model, tok, args, limit=None):
    docs = ifeval_docs()[:limit]
    process = ifeval_scorer()
    gens = generate(model, tok, [chat(tok, d["prompt"]) for d in docs], 1024, GREEDY, args.batch, args.seed)
    recs, agg = [], Counter()
    for d, (ids, trunc) in zip(docs, gens):
        text = tok.decode(ids, skip_special_tokens=True)
        r = process(d, [text])
        agg["prompt_strict"] += r["prompt_level_strict_acc"]
        agg["prompt_loose"] += r["prompt_level_loose_acc"]
        agg["inst_strict"] += sum(r["inst_level_strict_acc"]); agg["inst_loose"] += sum(r["inst_level_loose_acc"])
        agg["inst_n"] += len(r["inst_level_strict_acc"])
        recs.append(dict(key=d["key"], truncated=trunc, loop=trunc and is_loop(ids), length=len(ids),
                         cjk=bool(CJK.search(text)), strict=r["prompt_level_strict_acc"], completion=text))
    n = len(docs)
    summ = dict(prompt_strict=agg["prompt_strict"] / n, prompt_loose=agg["prompt_loose"] / n,
                inst_strict=agg["inst_strict"] / agg["inst_n"], inst_loose=agg["inst_loose"] / agg["inst_n"],
                **symptoms(recs))
    return recs, summ


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL)
    p.add_argument("--name", required=True)
    p.add_argument("--suites", default="gsm8k,symbolic,ifeval")
    p.add_argument("--modes", default="greedy,sample")
    p.add_argument("--batch", type=int, default=64)  # 128 overflows into shared system RAM (WDDM) and runs ~7x slower
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--mem-frac", type=float, default=0.8)
    p.add_argument("--temperature", type=float, default=1.0, help="for the 'sample' mode (results keyed sample_T<t>)")
    args = p.parse_args()

    torch.cuda.set_per_process_memory_fraction(args.mem_frac)  # OOM loudly instead of spilling to system RAM
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16).cuda().eval()
    out = ROOT / "runs" / "eval" / args.name
    out.mkdir(parents=True, exist_ok=True)
    summ_path = out / "summary.json"
    summary = json.loads(summ_path.read_text()) if summ_path.exists() else {}
    summary["model"] = args.model

    jobs = []
    for suite in args.suites.split(","):
        if suite in ("gsm8k", "symbolic"):
            jobs += [(suite, m) for m in args.modes.split(",")]
        elif suite == "ifeval":
            jobs.append(("ifeval", "greedy"))
        elif suite == "thinkprobe":
            jobs.append(("thinkprobe", "think"))
    for suite, mode in jobs:
        t = time.time()
        print(f"[{suite}/{mode}]", flush=True)
        if suite == "ifeval":
            recs, s = run_ifeval(model, tok, args)
        elif suite == "thinkprobe":
            recs, s = run_math(model, tok, gsm8k_test()[:128], mode, args, thinking=True)
        else:
            recs, s = run_math(model, tok, gsm8k_test() if suite == "gsm8k" else symbolic(), mode, args)
        s["seconds"] = round(time.time() - t)
        if mode == "sample" and args.temperature != 1.0:
            mode = f"sample_T{args.temperature:g}"
        summary[f"{suite}_{mode}"] = s
        with gzip.open(out / f"{suite}_{mode}.jsonl.gz", "wt", encoding="utf-8") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
        summ_path.write_text(json.dumps(summary, indent=2))
        print(f"  {json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in s.items()})}", flush=True)


if __name__ == "__main__":
    main()
