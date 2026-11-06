"""Mine the agent's practice data: base-model generations on translated GSM8K-train prompts (de/ru/th) until about
100 prompts show Chinese leaking in, under a GPU-time cap.

    .venv/Scripts/python langmix/mine_practice.py                      # mining, then the Chinese + English prompts
    .venv/Scripts/python langmix/mine_practice.py --status             # counts and running mixing rate, no GPU
    .venv/Scripts/python langmix/mine_practice.py --langs fr es --per-lang 16 --zh 0 --en 0 \
        --out runs/langmix/practice_pilot                              # quick pilot of further languages
    .venv/Scripts/python langmix/mine_practice.py --limit-prompts 6 --max-new 64 --out runs/langmix/smoke/practice

Prompts are the native-language question only, no instruction (as check_multi.py and the MGSM eval suites);
sampling is check.py's (T=0.6, top_p=0.95, 2,048 new tokens), batches of 24, seed of a batch = 1000 + index of its
first prompt. The pool interleaves the languages (de, ru, th, de, ru, th, ...), each language in data.py's fixed
seed-0 order of Mathoctopus/GSM8KInstruct_Parallel (GSM8K-train translations; MGSM stays held out). A prompt counts as
mixing if its completion has `mix` (>= 1 Han character, metrics.mix_flag), the same flag the eval scores.

Stops when --target mixing prompts are found or the mining GPU time reaches --max-gpu-min (default 60), whichever
comes first. Then generates the base answers for --zh Chinese prompts (meta-math/GSM8K_zh train) and --en English
prompts (GSM8K train).

Outputs (runs/langmix/practice/):
- gen.jsonl      every generation (all languages; resumable, finished ids are skipped)
- practice.json  the agent's practice sets, rebuilt after every batch:
    mix    the mixing prompts (up to --target), with their completion
    clean  matched non-mixing prompts: for each mixing prompt, an unused prompt of the SAME language whose completion
           had no CJK at all, with the nearest completion length (same truncation status when possible)
    zh     Chinese prompts with their base completions (zh_reply, accuracy)
    en     English prompts with their base completions
"""
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # before torch
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import data as D  # noqa: E402
import metrics as M  # noqa: E402

OUT = C.RUNS / "practice"


def score(item, text, n_tok, trunc):
    pred, boxed = M.numeric_pred(text)
    r = dict(id=item["id"], question=item["question"], gold=item["gold"], lang=item["lang"], kind=item["kind"],
             scorer="numeric", length=n_tok, truncated=trunc, pred=pred, boxed=boxed,
             correct=M.numeric_equal(pred, item["gold"]))
    r.update(M.language_fields(text, item["lang"]))
    r["completion"] = text
    return r


def build_sets(out, target, langs):
    rows = C.read_jsonl(out / "gen.jsonl")
    tgt = [r for r in rows if r["lang"] in langs]
    mix = [r for r in tgt if r["mix"]][:target]
    pool = [r for r in tgt if not r["cjk_any"]]
    used, clean = set(), []
    for m in mix:
        cands = [r for r in pool if r["id"] not in used and r["lang"] == m["lang"]]
        if not cands:
            continue
        best = min(cands, key=lambda r: (r["truncated"] != m["truncated"], abs(r["length"] - m["length"])))
        used.add(best["id"])
        clean.append(best)
    sets = dict(mix=mix, clean=clean, zh=[r for r in rows if r["lang"] == "zh"],
                en=[r for r in rows if r["lang"] == "en"])
    (out / "practice.json").write_text(json.dumps(sets, ensure_ascii=False), encoding="utf-8")
    return sets


def load_practice(out=OUT):
    """The practice sets (dict mix / clean / zh / en -> list of items with the base completion). Used by agent_tools
    and by eval.py's practice_* suites."""
    return json.loads((Path(out) / "practice.json").read_text(encoding="utf-8"))


def status(rows, langs, target):
    tgt = [r for r in rows if r["lang"] in langs]
    n, k = len(tgt), sum(r["mix"] for r in tgt)
    gpu_min = sum(r.get("gpu_s", 0) for r in tgt) / 60
    per = {lg: f"{sum(r['mix'] for r in tgt if r['lang'] == lg)}/{sum(r['lang'] == lg for r in tgt)}" for lg in langs}
    msg = f"mining: {n} prompts, {k} mixing ({100 * k / max(n, 1):.1f}%) {per}; {gpu_min:.1f} GPU-min"
    if 0 < k < target:
        msg += f"; ~{(target - k) * gpu_min / k:.0f} more GPU-min to reach {target}"
    print(msg, flush=True)
    return k, gpu_min


def run(items, path, a, label, stop=lambda: False):
    """Generate for every item not yet in `path`, one GPU lock per batch; checks stop() before each batch."""
    done = {r["id"] for r in C.read_jsonl(path)}
    for start in range(0, len(items), a.batch):
        chunk = [it for it in items[start:start + a.batch] if it["id"] not in done]
        if not chunk:
            continue
        if stop():
            return
        gens, secs = C.generate_chunk(a.model_obj, a.tok, [C.chat_prompt(a.tok, C.user_text(it)) for it in chunk],
                                      C.SEED0 + start, a.max_new, f"langmix/mine_practice.py {label} {start}")
        C.append_jsonl(path, [score(it, *g) | dict(gpu_s=round(secs / len(chunk), 2)) for it, g in zip(chunk, gens)])
        print(f"  {label} batch at {start}: {len(chunk)} prompts in {secs:.0f}s GPU", flush=True)


def interleave(langs, per_lang):
    pools = {lg: D.load(D.POOLS[lg])[:per_lang] for lg in langs}
    return [pools[lg][i] for i in range(per_lang) for lg in langs if i < len(pools[lg])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.MODEL)
    ap.add_argument("--langs", nargs="+", default=["de", "ru", "th"])
    ap.add_argument("--target", type=int, default=100)
    ap.add_argument("--max-gpu-min", type=float, default=60)
    ap.add_argument("--per-lang", type=int, default=1000, help="pool size per language")
    ap.add_argument("--zh", type=int, default=48)
    ap.add_argument("--en", type=int, default=48)
    ap.add_argument("--batch", type=int, default=C.BATCH)
    ap.add_argument("--max-new", type=int, default=C.MAX_NEW)
    ap.add_argument("--limit-prompts", type=int, default=None, help="smoke test: mine only this many prompts")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "gen.jsonl"

    if a.status:
        status(C.read_jsonl(path), a.langs, a.target)
        if (out / "practice.json").exists():
            print({k: len(v) for k, v in load_practice(out).items()})
        return

    a.tok = C.load_tok(a.model)
    a.model_obj = C.load_model(a.model)
    pool = interleave(a.langs, a.per_lang)[: a.limit_prompts]

    def stop():
        k, gpu_min = status(C.read_jsonl(path), a.langs, a.target)
        build_sets(out, a.target, a.langs)
        return k >= a.target or gpu_min >= a.max_gpu_min

    run(pool, path, a, "mine", stop)
    stop()
    run(D.load("gsm8k_zh_train")[: a.zh], path, a, "zh")
    run(D.load("gsm8k_train")[: a.en], path, a, "en")
    sets = build_sets(out, a.target, a.langs)
    for k in ("zh", "en"):
        v = sets[k]
        if v:
            extra = f", zh_reply {sum(r['zh_reply'] for r in v) / len(v):.2f}" if k == "zh" else ""
            print(f"{k}: {len(v)} prompts, acc {sum(r['correct'] for r in v) / len(v):.2f}, "
                  f"mix {sum(r['mix'] for r in v) / len(v):.2f}{extra}")
    print({k: len(v) for k, v in sets.items()}, "->", out / "practice.json")


if __name__ == "__main__":
    main()
