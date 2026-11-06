"""Re-judge refusal-suite completions truncated to their first N tokens (no GPU), and adopt them as the arm's
refusal-suite results under the short-generation protocol.

  python refusal/truncate.py --model microsoft/Phi-3-mini-4k-instruct --arm base --tokens 64 [--write]

Greedy decoding is prefix-consistent, so the first N tokens of a 256-token completion equal an N-token run (up to batch
numerics). Prints the agreement between labels on the full and the truncated text, overall and per suite and class.
--write keeps the originals as <suite>_256.jsonl and rewrites <suite>.jsonl with the truncated completions, so
eval.py/summarize.py score this arm on the same footing as arms generated with --refusal-max-new N.
"""
import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import judge as J  # noqa: E402

SUITES = ("xstest_safe", "orhard", "xstest_unsafe", "ortoxic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--arm", default="base")
    ap.add_argument("--tokens", type=int, nargs="+", default=[64])
    ap.add_argument("--write", action="store_true", help="adopt the first --tokens value for this arm")
    a = ap.parse_args()
    tok = C.load_tok(a.model)
    d = C.RUNS / "eval" / C.slug(a.model) / a.arm
    report = {}
    for n in a.tokens:
        agree, tot, per = 0, 0, {}
        trans = Counter()
        for s in SUITES:
            src = d / f"{s}_256.jsonl" if (d / f"{s}_256.jsonl").exists() else d / f"{s}.jsonl"
            recs = C.read_jsonl(src)
            ok = 0
            for r in recs:
                ids = tok(r["completion"], add_special_tokens=False).input_ids
                short = tok.decode(ids[:n], skip_special_tokens=True)
                l_full, l_short = J.classify(r["completion"]), J.classify(short)
                ok += J.refused(l_full, "any") == J.refused(l_short, "any")
                trans[(l_full, l_short)] += 1
            per[s] = dict(n=len(recs), agree_any=round(ok / max(1, len(recs)), 4),
                          rate_full_text=round(J.rates([J.classify(r["completion"]) for r in recs])["any"], 4))
            agree += ok
            tot += len(recs)
        changed = {f"{x}->{y}": c for (x, y), c in trans.items() if x != y}
        report[n] = dict(agree_any=round(agree / tot, 4), n=tot, per_suite=per, label_changes=changed,
                         label_changes_total=sum(changed.values()))
        print(f"first {n} tokens vs full: binary refusal agreement {agree / tot:.4f} over {tot}; per suite "
              + ", ".join(f"{s} {v['agree_any']:.3f}" for s, v in per.items()) + f"; 3-class changes {changed}")
    C.jdump(report, d / "truncation_agreement.json")
    if a.write:
        n = a.tokens[0]
        for s in SUITES:
            p = d / f"{s}.jsonl"
            keep = d / f"{s}_256.jsonl"
            if not keep.exists():
                shutil.copy(p, keep)
            rows = []
            for r in C.read_jsonl(keep):
                ids = tok(r["completion"], add_special_tokens=False).input_ids
                short = tok.decode(ids[:n], skip_special_tokens=True)
                rows.append(dict(r, completion=short, n_tokens=min(len(ids), n),
                                 truncated=len(ids) >= n or r["truncated"], label=J.classify(short),
                                 truncated_from=r.get("n_tokens")))
            p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows), encoding="utf-8")
        cp = d / "config.json"
        cfg = C.jload(cp)
        cfg["refusal_max_new"] = n
        C.jdump(cfg, cp)
        print(f"rewrote {d} refusal suites at {n} tokens (originals in *_256.jsonl)")


if __name__ == "__main__":
    main()
