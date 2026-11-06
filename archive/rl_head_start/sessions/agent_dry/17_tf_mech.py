import sys, json, torch, re, random; sys.path.insert(0, "../../rl"); import agent_tools as T
import steerlib
mode = sys.argv[1]  # "edited" or "base_general"
GEN = [
 "The city council met on Tuesday to discuss the new budget. Members agreed to increase funding for public parks, while postponing a decision on road repairs until next month. Several residents spoke in favor of extending library hours.",
 "Dear Ms. Patel,\n\nThank you for your letter of March 3. I am happy to confirm that your order has shipped and should arrive within five business days. Please let me know if you have any questions.\n\nBest regards,\nTom",
 "Here are three tips for better sleep:\n1. Keep a regular schedule, even on weekends.\n2. Avoid screens for an hour before bed.\n3. Keep your bedroom cool, dark, and quiet.",
 "def fibonacci(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n\nprint(fibonacci(10))",
 "Photosynthesis is the process by which green plants use sunlight, water, and carbon dioxide to produce glucose and oxygen. It takes place mainly in the chloroplasts of leaf cells.",
 "Write a haiku about autumn.\n\nCrimson leaves drifting\nacross the quiet river\nthe year exhales slow",
 "Summary: The report finds that remote work increased productivity for most teams surveyed, but also notes challenges with onboarding new employees and maintaining informal communication.",
 "Q: What is the capital of Australia?\nA: The capital of Australia is Canberra, not Sydney as many people assume.",
]
W = T.base_weights()
if mode == "edited":
    T.save_edit(steerlib.make_steer(W, 4.0), note="anti-hedging steer beta 4 (final)")
    rows = json.load(open("work/gen_base_disc.json"))
    P = {p["id"]: p for p in T.practice()}
    MARK = re.compile(r"(Wait|Alternatively|However|Let's re-|Let us re-|re-read|Re-read|Re-evaluat|Let's reconsider|If we interpret|Let's assume|Assuming|Let's check|Let's verify|double-check|Let's look|Let's try|Another interpretation|Or maybe|Maybe|Perhaps|Hmm|Actually|But )")
    random.seed(0)
    tr = [r for r in rows if r["truncated"] and MARK.search(r["completion"])]
    fin = [r for r in rows if not r["truncated"]]
    random.shuffle(tr); random.shuffle(fin)
    sel = tr[32:] + fin[32:52]
    R = torch.load("work/tf_collect2.pt", weights_only=False)
    assert [r["id"] for r in sel] == [it["id"] for it in R["items"]]
    tot = sum(len(it["ids"]) for it in R["items"])
    out = T.teacher_forced([r["completion"] for r in sel], layers=[], chat_question=[P[r["id"]]["question"] for r in sel])
    out_g = None
    stats = dict(trunc=[], fin=[], D=[], trunc_early=[], trunc_late=[], fin_correct=[])
    for it, o in zip(R["items"], out):
        assert o["ids"].tolist() == it["ids"]
        d = (o["logprobs"].float() - it["logprobs"].float())[it["s"] - 1:]
        stats["trunc" if it["truncated"] else "fin"].append(float(d.mean()))
        if not it["truncated"] and it["correct"]: stats["fin_correct"].append(float(d.mean()))
        if it["truncated"]:
            stats["trunc_early"].append(float(d[:150].mean())); stats["trunc_late"].append(float(d[-150:].mean()))
        for p in it["D"]: stats["D"].append(float(o["logprobs"][p] - it["logprobs"][p]))
    for k, v in stats.items():
        v = torch.tensor(v); print(f"{k:12} mean dlogprob/token {v.mean():+.4f} +- {v.std()/len(v)**0.5:.4f} (n={len(v)})")
    json.dump(stats, open("work/tf_mech_stats.json", "w"))
