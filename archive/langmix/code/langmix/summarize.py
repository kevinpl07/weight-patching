"""Pass-bar table for the language-mixing experiment, de/ru/th framing (orchestrator's adaptation of
PLAN_2026-11-04_practical_edits.md, section 2).

    .venv/Scripts/python langmix/summarize.py [--eval-root runs/langmix/eval] [--base base]

Reads runs/langmix/eval/<arm>/summary.json and the per-item jsonl files. Arm naming convention:
base, instr, ban, steer*, s<k>_agent, s<k>_rand<seed>_<mode>, script_* (anything else is listed too).

Pass bar (relative to the base arm):
  B1  de/ru/th mixing (>= 1 Han character, pooled over the 360 items) <= 1/3 of base           (plain threshold)
  B2  de+ru accuracy (pooled, 240 items; Thai is at the floor, 3%)      margin 3 points          (non-inferiority)
  B3  MGSM-en accuracy                                                   margin 3 points          (non-inferiority)
  B4  MGSM-zh accuracy, margin 3 points, AND Chinese-reply rate, margin 5 points               (non-inferiority)
  B5  IFEval prompt-level strict accuracy                                margin 2 points          (non-inferiority)
  B6  (agent arms) every matched random edit s<k>_rand* fails at least one of B1-B5
Non-inferiority (orchestrator's decision; deviation from the plan, because bars of 2-5 points at n = 100-240 are inside
sampling noise): an arm FAILS B2-B5 only if (a) its point estimate is below base by more than the margin AND (b) the
drop is significant: one-sided unpaired two-proportion z-test (pooled variance) of arm < base, p < 0.05.
The table prints the raw differences with unpaired 95% Wald intervals.
A session passes if B1-B6 hold; the experiment succeeds if 2 of 3 sessions pass. The base block prints 95% Wilson
intervals for every base rate. Missing suites show "-" and make the verdict "incomplete" (a random arm that already
fails a bar, e.g. B1 from the mixing suites alone, counts as failing).
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

TARGETS = ("mgsm_de", "mgsm_ru", "mgsm_th")
SUITES = TARGETS + ("mgsm_en", "mgsm_zh", "ifeval")


def wilson(k, n, z=1.96):
    if not n:
        return None, None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def diff_test(arm_rows, base_rows, key):
    """(difference arm - base, 95% Wald half-width, one-sided p of 'arm < base') for a binary field, unpaired."""
    if not arm_rows or not base_rows:
        return None, None, None
    n1, n0 = len(arm_rows), len(base_rows)
    p1, p0 = rate(arm_rows, key), rate(base_rows, key)
    hw = 1.96 * math.sqrt(p1 * (1 - p1) / n1 + p0 * (1 - p0) / n0)
    pp = (p1 * n1 + p0 * n0) / (n1 + n0)
    se = math.sqrt(pp * (1 - pp) * (1 / n1 + 1 / n0))
    pval = 0.5 * math.erfc(((p0 - p1) / se) / math.sqrt(2)) if se > 0 else (0.0 if p1 < p0 else 1.0)
    return p1 - p0, hw, pval


def noninf_ok(arm_rows, base_rows, key, margin):
    """True = passes (not significantly worse by more than the margin); None if missing."""
    d, _, pval = diff_test(arm_rows, base_rows, key)
    if d is None:
        return None
    return not (d < -margin and pval < 0.05)


def paired(a, b, key):
    ids = sorted(set(a) & set(b))
    if not ids:
        return None, None
    d = [float(a[i].get(key) or 0) - float(b[i].get(key) or 0) for i in ids]
    m = sum(d) / len(d)
    se = math.sqrt(sum((x - m) ** 2 for x in d) / max(len(d) - 1, 1) / len(d))
    return m, se


def rate(rows, key):
    return sum(bool(r.get(key)) for r in rows) / len(rows) if rows else None


def fmt(x, nd=1):
    return "-" if x is None else f"{100 * x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-root", default=str(C.RUNS / "eval"))
    ap.add_argument("--base", default="base")
    a = ap.parse_args()
    root = Path(a.eval_root)
    arms = sorted(p.parent.name for p in root.glob("*/summary.json"))
    if a.base not in arms:
        sys.exit(f"no base arm in {root}")
    arms.sort(key=lambda n: (n != a.base, not n.startswith(("instr", "sys", "ban", "steer")), n))
    R = {arm: {s: {r["id"]: r for r in C.read_jsonl(root / arm / f"{s}.jsonl")} for s in SUITES} for arm in arms}
    S = {arm: json.loads((root / arm / "summary.json").read_text()) for arm in arms}
    RB = R[a.base]

    def pooled(arm):
        out = {}
        for s in TARGETS:
            out.update(R[arm][s])
        return out

    PB = pooled(a.base)

    # ---- base rates with 95% Wilson intervals
    print(f"Base arm '{a.base}' (95% Wilson CI):")
    for s in SUITES:
        rows = list(RB[s].values())
        if not rows:
            continue
        keys = {"ifeval": ["strict"], "mgsm_zh": ["correct", "zh_reply", "mix"]}.get(s, ["correct", "mix", "han_run",
                                                                                         "han_answer", "reply_target"])
        parts = []
        for k in keys:
            n, c = len(rows), sum(bool(r.get(k)) for r in rows)
            lo, hi = wilson(c, n)
            parts.append(f"{k} {c}/{n} = {fmt(c / n)}% [{fmt(lo)}, {fmt(hi)}]")
        tls = [r["think_lang_share"] for r in rows if r.get("think_lang_share") is not None]
        if tls:
            parts.append(f"think in prompt language {fmt(sum(tls) / len(tls))}%")
        print(f"  {s}: " + "; ".join(parts))
    pr = list(PB.values())
    if pr:
        c = sum(r["mix"] for r in pr)
        lo, hi = wilson(c, len(pr))
        print(f"  de+ru+th pooled mixing {c}/{len(pr)} = {fmt(c / len(pr))}% [{fmt(lo)}, {fmt(hi)}] -> B1 needs <= "
              f"{fmt(c / len(pr) / 3)}% (<= {c // 3} of {len(pr)})")

    # ---- per-arm table
    def ci(d, hw):
        return "-" if d is None else f"{100 * d:+.1f} ±{100 * hw:.1f}"

    def rows_of(arm, suites):
        return [r for s in suites for r in R[arm][s].values()]

    print("\n| arm | mix de | mix ru | mix th | mix pooled | han in answer | acc de | acc ru | acc th | acc de+ru (Δ ±95%) "
          "| acc en (Δ) | mix en | acc zh (Δ) | zh-reply (Δ) | IFEval (Δ) | KL | B1 | B2 | B3 | B4 | B5 |")
    print("|" + "---|" * 21)
    verdict = {}
    m0 = rate(pr, "mix")
    B = {k: rows_of(a.base, v) for k, v in dict(dr=("mgsm_de", "mgsm_ru"), en=("mgsm_en",), zh=("mgsm_zh",),
                                                  ife=("ifeval",)).items()}
    for arm in arms:
        p = rows_of(arm, TARGETS)
        complete = all(len(R[arm][s]) == len(RB[s]) for s in TARGETS)
        mix = rate(p, "mix") if complete else None
        A = {k: rows_of(arm, v) for k, v in dict(dr=("mgsm_de", "mgsm_ru"), en=("mgsm_en",), zh=("mgsm_zh",),
                                                   ife=("ifeval",)).items()}
        full_dr = all(len(R[arm][s]) == len(RB[s]) for s in ("mgsm_de", "mgsm_ru"))
        b1 = None if mix is None else mix <= m0 / 3
        b2 = noninf_ok(A["dr"], B["dr"], "correct", 0.03) if full_dr else None
        b3 = noninf_ok(A["en"], B["en"], "correct", 0.03)
        z1, z2 = noninf_ok(A["zh"], B["zh"], "correct", 0.03), noninf_ok(A["zh"], B["zh"], "zh_reply", 0.05)
        b4 = None if z1 is None else (z1 and z2)
        b5 = noninf_ok(A["ife"], B["ife"], "strict", 0.02)
        verdict[arm] = (b1, b2, b3, b4, b5)
        yn = lambda b: "-" if b is None else ("yes" if b else "NO")  # noqa: E731
        kl = S[arm].get("chat", {}).get("kl")
        per_mix = [fmt(rate(list(R[arm][s].values()), "mix")) for s in TARGETS]
        per_acc = [fmt(rate(list(R[arm][s].values()), "correct")) for s in TARGETS]
        dd = lambda k, key: ci(*diff_test(A[k], B[k], key)[:2])  # noqa: E731
        print(f"| {arm} | {' | '.join(per_mix)} | {fmt(mix)} | {fmt(rate(p, 'han_answer'))} | {' | '.join(per_acc)} | "
              f"{fmt(rate(A['dr'], 'correct'))} ({dd('dr', 'correct')}) | {fmt(rate(A['en'], 'correct'))} "
              f"({dd('en', 'correct')}) | {fmt(rate(A['en'], 'mix'))} | {fmt(rate(A['zh'], 'correct'))} "
              f"({dd('zh', 'correct')}) | {fmt(rate(A['zh'], 'zh_reply'))} ({dd('zh', 'zh_reply')}) | "
              f"{fmt(rate(A['ife'], 'strict'))} ({dd('ife', 'strict')}) | {'-' if kl is None else f'{kl:.4f}'} | "
              + " | ".join(yn(b) for b in verdict[arm]) + " |")

    print("\nSessions:")
    passes = 0
    for arm in arms:
        m = re.fullmatch(r"s(\d+)_agent", arm)
        if not m:
            continue
        k = m.group(1)
        bars = verdict[arm]
        rands = [r for r in arms if r.startswith(f"s{k}_rand")]
        r_fail = [any(b is False for b in verdict[r]) for r in rands]
        b6 = bool(rands) and all(r_fail)
        v = "incomplete" if any(b is None for b in bars) or not rands else (
            "PASS" if all(bars) and b6 else "fail")
        passes += v == "PASS"
        print(f"  s{k}: B1-B5 {['-' if b is None else int(b) for b in bars]}  random {rands} fail: {r_fail} -> {v}")
    print(f"\n{passes} session(s) pass; the experiment succeeds at 2 of 3.")


if __name__ == "__main__":
    main()
