"""Tables + example picker for results/signoff/NOTES.md (CPU only).

    python -m tasks._signoff_summary table      # main arms
    python -m tasks._signoff_summary sweep      # sweep mean +- sd per (method, n)
    python -m tasks._signoff_summary examples   # candidate before/after examples
"""
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from tasks._signoff_detect import signoff  # noqa: E402

RES = ROOT / "results" / "signoff"
ARMS = ["base", "prompt", "gate", "random", "lora"]


def table():
    cols = ["signoff", "signoff_practice_phrase", "signoff_heldout_phrase_only", "content_kept", "content_identical",
            "legit_kept", "appendix", "dangling_rule", "truncated", "mean_chars"]
    print("| arm | entries / params | " + " | ".join(cols) + " | MMLU-200 | chat KL |")
    print("|" + "---|" * (len(cols) + 4))
    for a in ARMS:
        p = RES / f"{a}.json"
        if not p.exists():
            continue
        r = json.loads(p.read_text(encoding="utf-8"))
        m, g, c = r["metrics"], r["guards"], r["cost"]
        size = c.get("entries_changed") or c.get("trainable_params") or 0
        print(f"| {a} | {size:,} | " + " | ".join(str(m.get(k, "")) for k in cols)
              + f" | {g['mmlu200'] if g['mmlu200'] is not None else '-'} | {g['chat_kl']} |")


def sweep():
    agg = defaultdict(list)
    for p in sorted((RES / "sweep").glob("*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        meth = p.name.split("_")[0]
        agg[(meth, r["n_practice"])].append(r)
    print("| method | n practice | runs | sign-off % | content kept % | legit kept % | chat KL |")
    print("|---|---|---|---|---|---|---|")
    for (meth, n), rs in sorted(agg.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        def ms(k, src="metrics"):
            v = [r[src][k] for r in rs if r[src].get(k) is not None]
            if not v:
                return "-"
            return f"{st.mean(v):.1f} ± {st.stdev(v):.1f}" if len(v) > 1 else f"{v[0]:.1f}"
        kl = [r["guards"]["chat_kl"] for r in rs]
        print(f"| {meth} | {n} | {len(rs)} | {ms('signoff')} | {ms('content_kept')} | {ms('legit_kept')} | "
              f"{st.mean(kl):.4f} |")


def examples(k=40):
    items = {x["id"]: x for x in map(json.loads, open(RES / "items.jsonl", encoding="utf-8"))}
    outs = {a: {r["id"]: r["text"] for r in map(json.loads, open(RES / "outputs" / f"{a}.jsonl", encoding="utf-8"))}
            for a in ("gate", "lora", "prompt")}
    n = 0
    for i, x in items.items():
        if x["split"] != "heldout" or x.get("legit"):
            continue
        b = x["ref"]
        s = signoff(b)
        if not s["has"] or len(b) > 1500:
            continue
        g, lo = outs["gate"][i], outs["lora"][i]
        if signoff(g)["has"]:
            continue
        n += 1
        print("=" * 100)
        print("ID", i, "|", x["prompt"][:200].replace("\n", " "))
        for name, t in (("BASE", b), ("GATE", g), ("LORA", lo), ("PROMPT", outs["prompt"][i])):
            print(f"--- {name} ({len(t)} chars, signoff={signoff(t)['has']}):\n...{t[-300:]}")
        if n >= k:
            break


if __name__ == "__main__":
    {"table": table, "sweep": sweep, "examples": examples}[sys.argv[1]]()
