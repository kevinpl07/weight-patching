"""Viability check for experiment 2 (language mixing): does DeepSeek-R1-Distill-Qwen-1.5B mix Chinese into
English reasoning often enough (>= 8% of answers) to be worth editing?

    .venv/Scripts/python langmix/check.py --n 200 [--model deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B]

Settings (keep identical in the later eval so these rows double as the base arm on the same problems):
- MATH-500 (HuggingFaceH4/MATH-500, test split), problems in a fixed order (shuffled with seed 0), first --n.
- DeepSeek's recommended usage: no system prompt; user turn = problem + "\nPlease reason step by step, and put your
  final answer within \\boxed{}."; chat template with add_generation_prompt=True.
- Sampling T=0.6, top_p=0.95, max_new_tokens=2048, one sample per problem, seed = 1000 + batch index.
- Mixing: `cjk_any` = any CJK character in the completion; `cjk_run` = a run of >= 2 consecutive Han characters.

Takes the GPU lock per batch, so other jobs can interleave. Resumable: appends to the jsonl and skips done ids.
Output: runs/langmix/check/<model_tag>/math500.jsonl and summary.json
"""
import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from gpu_lock import gpu  # noqa: E402

CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
HAN_RUN = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]{2,}")
SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."
SAMPLING = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=0)


def last_boxed(text):
    i = text.rfind("\\boxed{")
    if i < 0:
        return None
    i += 7
    depth, j = 1, i
    while j < len(text) and depth:
        depth += {"{": 1, "}": -1}.get(text[j], 0)
        j += 1
    return text[i:j - 1] if depth == 0 else None


def norm(s):
    s = (s or "").replace(" ", "").replace("\\!", "").replace("dfrac", "frac").replace("tfrac", "frac")
    s = s.replace("\\left", "").replace("\\right", "").replace("^\\circ", "").replace("^{\\circ}", "")
    s = re.sub(r"\\text\{(.*?)\}", r"\1", s)
    return s.strip("$.").rstrip(".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--batch", type=int, default=50)
    ap.add_argument("--max_new", type=int, default=2048)
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    ds = list(load_dataset("HuggingFaceH4/MATH-500", split="test"))
    order = list(range(len(ds)))
    random.Random(0).shuffle(order)
    probs = [dict(id=ds[i]["unique_id"], problem=ds[i]["problem"], answer=ds[i]["answer"]) for i in order[:a.n]]

    out_dir = ROOT / "runs" / "langmix" / "check" / a.model.split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "math500.jsonl"
    done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()
    todo = [p for p in probs if p["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).eval()  # stays on CPU between batches

    for b in range(0, len(todo), a.batch):
        chunk = todo[b:b + a.batch]
        prompts = [tok.apply_chat_template([dict(role="user", content=p["problem"] + SUFFIX)], tokenize=False,
                                           add_generation_prompt=True) for p in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        t = time.time()
        with gpu(f"langmix/check.py batch {b // a.batch}"):
            torch.cuda.set_per_process_memory_fraction(0.85)
            model.cuda()
            torch.manual_seed(1000 + len(done) + b)
            with torch.no_grad():
                gen = model.generate(**{k: v.cuda() for k, v in enc.items()}, max_new_tokens=a.max_new,
                                     pad_token_id=tok.pad_token_id, **SAMPLING)
            gen = gen[:, enc.input_ids.shape[1]:].cpu()
            model.cpu()
        with open(out, "a", encoding="utf-8") as f:
            for p, ids in zip(chunk, gen):
                ids = ids.tolist()
                n_tok = next((i for i, x in enumerate(ids) if x in (tok.pad_token_id, tok.eos_token_id)), len(ids))
                text = tok.decode(ids[:n_tok], skip_special_tokens=True)
                pred = last_boxed(text)
                f.write(json.dumps(dict(id=p["id"], answer=p["answer"], pred=pred,
                                        correct=pred is not None and norm(pred) == norm(p["answer"]),
                                        truncated=n_tok >= a.max_new, length=n_tok,
                                        cjk_any=bool(CJK.search(text)), cjk_run=bool(HAN_RUN.search(text)),
                                        completion=text), ensure_ascii=False) + "\n")
        print(f"batch {b // a.batch}: {len(chunk)} problems in {time.time() - t:.0f}s", flush=True)

    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    n = len(rows)
    s = dict(model=a.model, n=n, **{k: round(sum(r[k] for r in rows) / n, 4)
                                    for k in ("cjk_any", "cjk_run", "correct", "truncated")},
             mean_len=round(sum(r["length"] for r in rows) / n))
    (out_dir / "summary.json").write_text(json.dumps(s, indent=1))
    print(json.dumps(s, indent=1))
    print("PASS (>= 8% mixing)" if s["cjk_run"] >= 0.08 else "FAIL (< 8% mixing)")


if __name__ == "__main__":
    main()
