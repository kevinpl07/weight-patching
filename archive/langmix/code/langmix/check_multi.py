"""Viability check for the multilingual framing of experiment 2: asked in language X, does the model think and/or
answer in a different language (English or Chinese)?

    .venv/Scripts/python langmix/check_multi.py --per_lang 16 [--model ...] [--langs de ru th sw bn te]

MGSM (juletxara/mgsm, test split), first --per_lang problems per language after a seed-0 shuffle. User turn = the
native-language question only (no English instruction, so nothing nudges the language). Sampling as check.py
(T=0.6, top_p=0.95), max_new_tokens 2048. Per answer we record, for the thinking part (before </think>) and the final
answer (after it): langdetect language on text with math/digits stripped, share of letters in the target script
(for non-Latin languages), and whether a Han run appears. Lock per batch, resumable.
Output: runs/langmix/check_multi/<model_tag>/mgsm.jsonl and summary.json
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
sys.path.insert(0, str(ROOT / "langmix"))
from gpu_lock import gpu  # noqa: E402
from check import HAN_RUN, SAMPLING  # noqa: E402

SCRIPT = {"ru": r"[Ѐ-ӿ]", "th": r"[฀-๿]", "bn": r"[ঀ-৿]", "te": r"[ఀ-౿]",
          "ja": r"[぀-ヿ一-鿿]", "zh": r"[一-鿿]"}


def clean(t):
    t = re.sub(r"\$[^$]*\$|\\\[.*?\\\]|\\\(.*?\\\)|\\boxed\{[^}]*\}", " ", t, flags=re.S)
    return re.sub(r"[0-9=+\-*/^()\[\]{}\\_.,:;%$]", " ", t)


def lang_of(t):
    from langdetect import DetectorFactory, detect
    DetectorFactory.seed = 0
    t = clean(t)
    if len(re.findall(r"\w", t)) < 20:
        return None
    try:
        return detect(t)
    except Exception:
        return None


def script_share(t, lang):
    if lang not in SCRIPT:
        return None
    letters = re.findall(r"[^\W\d_]", clean(t))
    return round(len(re.findall(SCRIPT[lang], "".join(letters))) / len(letters), 3) if letters else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    ap.add_argument("--langs", nargs="+", default=["de", "ru", "th", "sw", "bn", "te"])
    ap.add_argument("--per_lang", type=int, default=16)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--max_new", type=int, default=2048)
    a = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoModelForCausalLM, AutoTokenizer
    probs = []
    for lg in a.langs:
        ds = list(load_dataset("juletxara/mgsm", lg, split="test"))
        order = list(range(len(ds)))
        random.Random(0).shuffle(order)
        probs += [dict(id=f"{lg}-{i}", lang=lg, question=ds[i]["question"], answer=str(ds[i]["answer_number"]))
                  for i in order[:a.per_lang]]

    out_dir = ROOT / "runs" / "langmix" / "check_multi" / a.model.replace("\\", "/").rstrip("/").split("/")[-1]
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / "mgsm.jsonl"
    done = {json.loads(l)["id"] for l in out.read_text(encoding="utf-8").splitlines()} if out.exists() else set()
    todo = [p for p in probs if p["id"] not in done]
    print(f"{len(done)} done, {len(todo)} to go", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).eval()
    for b in range(0, len(todo), a.batch):
        chunk = todo[b:b + a.batch]
        prompts = [tok.apply_chat_template([dict(role="user", content=p["question"])], tokenize=False,
                                           add_generation_prompt=True) for p in chunk]
        enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False)
        t = time.time()
        with gpu(f"langmix/check_multi.py batch {b // a.batch}"):
            torch.cuda.set_per_process_memory_fraction(0.85)
            model.cuda()
            torch.manual_seed(2000 + len(done) + b)
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
                think, sep, ans = text.partition("</think>")
                if not sep:
                    ans = ""
                nums = re.findall(r"-?\d[\d,]*\.?\d*", ans or think)
                pred = nums[-1].replace(",", "").rstrip(".") if nums else None
                f.write(json.dumps(dict(
                    id=p["id"], lang=p["lang"], answer=p["answer"], pred=pred,
                    correct=pred is not None and pred.rstrip("0").rstrip(".") == p["answer"].rstrip("0").rstrip("."),
                    truncated=n_tok >= a.max_new, closed_think=bool(sep), length=n_tok,
                    think_lang=lang_of(think), ans_lang=lang_of(ans) if ans else None,
                    think_script=script_share(think, p["lang"]), ans_script=script_share(ans, p["lang"]) if ans else None,
                    han_think=bool(HAN_RUN.search(think)), han_ans=bool(HAN_RUN.search(ans)),
                    completion=text), ensure_ascii=False) + "\n")
        print(f"batch {b // a.batch}: {len(chunk)} in {time.time() - t:.0f}s", flush=True)

    rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines()]
    summ = {}
    for lg in a.langs + ["all"]:
        r = [x for x in rows if lg == "all" or x["lang"] == lg]
        if not r:
            continue
        n = len(r)
        a_rows = [x for x in r if x["ans_lang"]]
        summ[lg] = dict(
            n=n, acc=round(sum(x["correct"] for x in r) / n, 3), trunc=round(sum(x["truncated"] for x in r) / n, 3),
            think_wrong=round(sum(x["think_lang"] not in (None, x["lang"]) for x in r) / n, 3),
            ans_wrong=round(sum(x["ans_lang"] != x["lang"] for x in a_rows) / max(len(a_rows), 1), 3),
            n_ans=len(a_rows), han_any=round(sum(x["han_think"] or x["han_ans"] for x in r) / n, 3),
            think_langs=dict(sorted({k: sum(x["think_lang"] == k for x in r) for k in {x["think_lang"] for x in r}
                                     }.items(), key=lambda kv: -kv[1])[:4]))
    (out_dir / "summary.json").write_text(json.dumps(summ, indent=1, default=str))
    print(json.dumps(summ, indent=1, default=str))


if __name__ == "__main__":
    main()
