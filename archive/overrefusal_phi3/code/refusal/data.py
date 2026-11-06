"""Data for the over-refusal experiment. Splits are fixed (seed 0) and cached as JSON in runs/refusal/data/.

  python refusal/data.py prep                       # CPU: download/load, split, write held_out.json + practice_pool.json
  python refusal/data.py label --model <hf id>      # GPU: base-model answers on the practice pool -> practice.json

Sources (verified 2026-11-04):
- XSTest: `Paul/XSTest` (Paul Röttger's own upload, ungated; 450 rows: id, prompt, type, label safe/unsafe, focus,
  note; 250 safe, 200 unsafe). `walledai/XSTest` named in the brief is gated for this account; same 450 prompts.
- OR-Bench `bench-llm/or-bench`, configs `or-bench-hard-1k` (1,319), `or-bench-toxic` (655), `or-bench-80k` (80,359);
  fields prompt, category; single split "train".
- MMLU `cais/mmlu` config "all", split test (14,042).
- IFEval `google/IFEval` (541), scored by lm_eval's checker (eval.py).
- KL prompts: `HuggingFaceH4/no_robots` split test (500; ungated, 11 MB). Field `prompt` (the first user turn),
  prompts <= 2,000 characters, 200 sampled with seed 0. Any system message in `messages` is ignored.

Held-out (the agent never sees these): XSTest safe 250 + unsafe 200 (all), OR-Hard 300, OR-Toxic 200, MMLU 500
(round-robin over the 57 subjects in shuffled order), KL 200.
Practice pool (disjoint from held-out, exact-duplicate prompts removed across all OR-Bench splits):
  hard  = all remaining OR-Hard prompts (1,019, in the shuffled order after the 300 held-out); labelled lazily
  easy  = 100 OR-Bench-80k prompts that are not in hard-1k or toxic ("easy" benign; same generator, not hard-filtered)
  toxic = 100 OR-Toxic prompts (next 100 after the 200 held-out)
`label` runs the base model (greedy, 256 tokens, no system prompt; same settings as eval.py) on the easy and toxic
pools and on the hard pool in order, chunk by chunk, until at least --min-hard (400) hard prompts are labelled AND
100 of them are over-refused (or the hard pool runs out). Then it picks:
  over_refused (100): hard-pool prompts judged full_refusal or partial_refusal (the "any" metric), in pool order
  complied     (100): hard-pool prompts judged full_compliance in pool order, then easy-pool ones if needed
  harmful      (100): the toxic pool (whatever the base model did; its label is stored)
If all 1,019 hard prompts give fewer than 100 over-refused prompts it says so and uses what there is.
Resumable: already-answered prompts are never regenerated.
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import judge as J  # noqa: E402

SEED = 0


def _ds(name, config=None, split="train"):
    from datasets import load_dataset
    return load_dataset(name, config, split=split)


def _shuffled(rows, tag):
    rows = list(rows)
    random.Random(f"{SEED}-{tag}").shuffle(rows)
    return rows


def _dedupe(rows, seen):
    out = []
    for r in rows:
        k = r["prompt"].strip().lower()
        if k not in seen:
            seen.add(k)
            out.append(r)
    return out


def prep(a):
    xs = _ds("Paul/XSTest")
    xstest = [dict(uid=f"xs-{r['id']}", prompt=r["prompt"], label=r["label"], type=r["type"], focus=r["focus"])
              for r in xs]
    xs_safe = [r for r in xstest if r["label"] == "safe"]
    xs_unsafe = [r for r in xstest if r["label"] == "unsafe"]
    assert len(xs_safe) == 250 and len(xs_unsafe) == 200, (len(xs_safe), len(xs_unsafe))

    seen = set()
    hard = _dedupe(_shuffled([dict(uid=f"orh-{i}", prompt=r["prompt"], category=r["category"])
                              for i, r in enumerate(_ds("bench-llm/or-bench", "or-bench-hard-1k"))], "hard"), seen)
    toxic = _dedupe(_shuffled([dict(uid=f"ort-{i}", prompt=r["prompt"], category=r["category"])
                               for i, r in enumerate(_ds("bench-llm/or-bench", "or-bench-toxic"))], "toxic"), seen)
    big = _ds("bench-llm/or-bench", "or-bench-80k")
    idx = list(range(len(big)))
    random.Random(f"{SEED}-80k").shuffle(idx)
    easy = _dedupe([dict(uid=f"or80k-{i}", prompt=big[i]["prompt"], category=big[i]["category"])
                    for i in idx[:5000]], seen)[:a.pool_easy]
    n_h = 300

    mm = _ds("cais/mmlu", "all", "test")
    by_subj = {}
    for i, r in enumerate(mm):
        by_subj.setdefault(r["subject"], []).append(dict(uid=f"mmlu-{i}", question=r["question"], subject=r["subject"],
                                                         choices=list(r["choices"]), answer=int(r["answer"])))
    subj = _shuffled(sorted(by_subj), "mmlu-subjects")
    pools = {s: _shuffled(by_subj[s], f"mmlu-{s}") for s in subj}
    mmlu, k = [], 0
    while len(mmlu) < 500:
        for s in subj:
            if k < len(pools[s]) and len(mmlu) < 500:
                mmlu.append(pools[s][k])
        k += 1

    nr = [dict(uid=f"nr-{r['prompt_id'][:12]}", prompt=r["prompt"], category=r["category"])
          for r in _ds("HuggingFaceH4/no_robots", split="test") if len(r["prompt"]) <= 2000]
    kl = _shuffled(nr, "kl")[:200]

    held = dict(xstest_safe=xs_safe, xstest_unsafe=xs_unsafe, orhard=hard[:n_h], ortoxic=toxic[:200], mmlu=mmlu, kl=kl)
    pool = dict(hard=hard[n_h:], easy=easy, toxic=toxic[200:300])
    held_prompts = {r["prompt"].strip().lower() for s in ("xstest_safe", "xstest_unsafe", "orhard", "ortoxic")
                    for r in held[s]}
    for k_, rows in pool.items():
        assert not ({r["prompt"].strip().lower() for r in rows} & held_prompts), f"pool {k_} overlaps held-out"
    C.jdump(held, C.DATA / "held_out.json")
    C.jdump(pool, C.DATA / "practice_pool.json")
    print("held-out:", {k: len(v) for k, v in held.items()}, "(+ IFEval 541)")
    print("practice pool:", {k: len(v) for k, v in pool.items()})
    print("MMLU subjects covered:", len({r['subject'] for r in mmlu}))


def label(a):
    pool = C.jload(C.DATA / "practice_pool.json")
    out_dir = C.DATA / C.slug(a.model)
    path = out_dir / "pool_labels.jsonl"
    lim = a.limit
    if lim:
        print(f"SMOKE: only the first {lim} prompts of each pool; rerun without --limit for the real practice set")
    fixed = [dict(r, pool=k) for k in ("easy", "toxic") for r in pool[k][:lim]]
    hard_all = [dict(r, pool="hard") for r in pool["hard"][:lim]]
    tok = model = None

    def run(todo):
        nonlocal tok, model
        if model is None:
            tok, model = C.load(a.model)
        for s in range(0, len(todo), a.chunk):
            chunk = todo[s:s + a.chunk]
            with C.on_gpu(model, f"refusal label {C.slug(a.model)} {s}"):
                gens = C.generate(model, tok, [C.chat(tok, r["prompt"]) for r in chunk], 256, a.batch)
            C.append_jsonl(path, [dict(uid=r["uid"], pool=r["pool"], prompt=r["prompt"], completion=g[0],
                                       n_tokens=g[1], truncated=g[2], label=J.classify(g[0]))
                                  for r, g in zip(chunk, gens)])

    done = {r["uid"] for r in C.read_jsonl(path)}
    run([r for r in fixed if r["uid"] not in done])
    n_hard = 0
    while True:
        lab = {r["uid"]: J.classify(r["completion"]) for r in C.read_jsonl(path)}
        prefix = []
        for r in hard_all:  # labelled prefix of the hard pool
            if r["uid"] not in lab:
                break
            prefix.append(r)
        n_over = sum(J.refused(lab[r["uid"]], "any") for r in prefix)
        print(f"hard pool: {len(prefix)} labelled, {n_over} over-refused", flush=True)
        if (len(prefix) >= a.min_hard and n_over >= 100) or len(prefix) == len(hard_all):
            break
        need = max(a.min_hard - len(prefix), a.chunk)
        run(hard_all[len(prefix):len(prefix) + need])
    n_hard = len(prefix)
    items = [r for r in hard_all[:n_hard]] + fixed
    lab = {r["uid"]: r for r in C.read_jsonl(path)}
    rows = [dict(r, base_completion=lab[r["uid"]]["completion"], base_label=J.classify(lab[r["uid"]]["completion"]))
            for r in items]
    hard = [r for r in rows if r["pool"] == "hard"]
    over = [r for r in hard if J.refused(r["base_label"], "any")][:100]
    comp = ([r for r in hard if r["base_label"] == "full_compliance"]
            + [r for r in rows if r["pool"] == "easy" and r["base_label"] == "full_compliance"])[:100]
    harm = [r for r in rows if r["pool"] == "toxic"][:100]
    practice = []
    for grp, rs in (("over_refused", over), ("complied", comp), ("harmful", harm)):
        for r in rs:
            practice.append(dict(id=len(practice), uid=r["uid"], group=grp, prompt=r["prompt"],
                                 category=r.get("category"), source=r["pool"], base_completion=r["base_completion"],
                                 base_label=r["base_label"]))
    C.jdump(practice, out_dir / "practice.json")
    print(f"hard prompts labelled: {n_hard} of {len(pool['hard'])}")
    for k in ("hard", "easy", "toxic"):
        print(f"pool {k}: {J.rates([r['base_label'] for r in rows if r['pool'] == k])}")
    print(f"practice: over_refused {len(over)}, complied {len(comp)} "
          f"({sum(r['pool'] == 'hard' for r in comp)} from hard), harmful {len(harm)} "
          f"(base refusal rate, any {sum(J.refused(r['base_label'], 'any') for r in harm) / max(1, len(harm)):.2f})")
    if len(over) < 100:
        print(f"WARNING: only {len(over)} over-refused prompts in {n_hard} labelled hard-pool prompts.")
    print(f"GPU seconds this run: {C.on_gpu.seconds:.0f}; wrote {out_dir / 'practice.json'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "label"])
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--min-hard", type=int, default=400, help="label at least this many OR-Hard pool prompts")
    ap.add_argument("--pool-easy", type=int, default=100)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--chunk", type=int, default=128)
    ap.add_argument("--limit", type=int, help="label only the first N prompts of each pool (smoke test)")
    a = ap.parse_args()
    {"prep": prep, "label": label}[a.cmd](a)


if __name__ == "__main__":
    main()
