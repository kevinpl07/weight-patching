import sys, json, re, math, collections
sys.path.insert(0, "../../rl"); import agent_tools as T
MARK = re.compile(r"(Wait|Alternatively|However|re-read|interpret)")
def paired(res):
    b, e = res["base"], res["edited"]
    for k in ("base", "edited"):
        s = T.summarize(res[k]); mk = sum(len(MARK.findall(r["completion"])) for r in res[k]) / len(res[k])
        fin = [r for r in res[k] if not r["truncated"]]
        print(f"{k:7} n={s['n']} acc {s['acc']:.3f} boxed {s['boxed']:.3f} trunc {s['trunc']:.3f} len {s['len']:.0f} markers/ans {mk:.2f} acc|finished {sum(r['correct'] for r in fin)/max(1,len(fin)):.3f}")
    # per-problem paired difference in accuracy
    pb = collections.defaultdict(list); pe = collections.defaultdict(list)
    for r in b: pb[r["id"]].append(r["correct"])
    for r in e: pe[r["id"]].append(r["correct"])
    d = [sum(pe[i]) / len(pe[i]) - sum(pb[i]) / len(pb[i]) for i in pb]
    m = sum(d) / len(d); sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5
    print(f"paired acc diff {m:+.3f} +- {sd/len(d)**0.5:.3f} (SE, {len(d)} problems)  z={m/(sd/len(d)**0.5+1e-9):.2f}")
    return m, sd / len(d) ** 0.5
if __name__ == "__main__":
    for f in sys.argv[1:]:
        print("==", f); paired(json.load(open(f)))
