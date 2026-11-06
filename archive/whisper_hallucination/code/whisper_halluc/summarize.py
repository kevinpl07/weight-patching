"""Build the pass-bar tables for experiment 3 from runs/whisper/eval/<arm>/summary.json.

  python whisper_halluc/summarize.py [--eval runs/whisper/eval] [--sessions 1,2,3] [--out runs/whisper]
  -> <out>/summary.md and <out>/summary.json

Arm names it understands (any other arm is listed without session verdicts):
  base, thresholds, thresholds_strict, vad, vad_thresholds, head_scan, head_scan_vad, agent<i>, agent<i>_vad,
  random<i>_same / random<i>_count (the matched random edits of session i), recipe<i> (post-hoc recipe script)

Per-arm / per-session pass bar (plan section 3, as revised 2026-11-04 by the orchestrator):
  - held-out non-speech hallucination falls by >= 50% relative to base (stricter version: to <= 1/3 of base)
  - padded-speech insertions do not rise: insertion rate <= base + 0.2 points (replaces "insertions halve")
  - clean WER (test-clean, 500) rises by <= 0.3 points
  - hard guards, speech in background (bg, 300) and test-other (other, 300): WER rises by <= 1.0 point and the share
    of fully dropped (empty) transcripts rises by <= 1 point
  - the session's random edit(s) improve hallucination by < 10% relative
Experiment succeeds if 2 of 3 sessions pass and at least one of:
  (a) agents beat thresholds by >= 15 points on the music+noise subset at equal WER cost (agent dWER <= thr dWER + 0.1)
  (b) agents beat the head scan by >= 10 relative points of hallucination drop at matched WER (dWER <= scan + 0.1)
  (c) edit + VAD cuts the non-speech that gets through VAD by >= 30% relative (agent<i>_vad vs vad)
"""
import argparse
import json
import re
from pathlib import Path

import common as C

ORDER = ["base", "thresholds", "thresholds_strict", "openai_thresholds", "vad", "vad_thresholds",
         "vad_thresholds_strict", "head_scan", "head_scan_vad", "head_scan_random"]


def load(eval_dir):
    arms = {}
    for p in sorted(Path(eval_dir).glob("*/summary.json")):
        s = json.loads(p.read_text())
        su = s.get("suites", {})
        cfg = s.get("config", {})
        row = dict(arm=p.parent.name, entries=cfg.get("edit_entries_changed", 0), limit=cfg.get("limit"))
        h = su.get("nonspeech", {}).get("hallucination", {})
        if h:
            row["halluc"] = h["overall"]["rate"]
            row["halluc_notag"] = h["overall"].get("rate_notag")
            for c in C.CATEGORIES:
                row[c] = h.get(c, {}).get("rate")
                row[c + "_notag"] = h.get(c, {}).get("rate_notag")
            n = {c: h.get(c, {}).get("n", 0) for c in C.CATEGORIES}
            mn = n["music"] + n["noise"]
            row["music_noise"] = ((h.get("music", {}).get("rate", 0) * n["music"] +
                                   h.get("noise", {}).get("rate", 0) * n["noise"]) / mn) if mn else None
        if "padded" in su:
            w = su["padded"]["words"]
            row.update(ins_rate=w["ins_rate"], any_ins=w["any_ins"], padded_wer=w["wer"])
            for pad, v in su["padded"].get("by_pad", {}).items():
                row[f"ins_{pad}"] = v["ins_rate"]
        for suite, pre in (("wer", ""), ("bg", "bg_"), ("other", "other_")):
            if suite in su:
                row[f"{pre}wer"] = su[suite]["words"]["wer"]
                row[f"{pre}dropped"] = su[suite]["words"].get("dropped")
        for snr, v in su.get("bg", {}).get("by_snr", {}).items():
            row[f"bg_wer_snr{snr}"] = v["wer"]
        for t, v in su.get("bg", {}).get("by_bg", {}).items():
            row[f"bg_wer_{t}"] = v["wer"]
        row["gpu_min"] = round(sum(v["seconds"] for v in s.get("gpu_seconds", {}).values()) / 60, 1)
        arms[row["arm"]] = row
    return arms


def rel_drop(x, b):
    return None if x is None or b in (None, 0) else 1 - x / b


def diff(r, b, k):
    return (r[k] - b[k]) if r.get(k) is not None and b.get(k) is not None else None


def cell(v, kind=None):
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if kind == "p":
        return f"{100 * v:.1f}"
    if kind == "d":
        return f"{v:+.2f}"
    if isinstance(v, float):
        return f"{v:.2f}"
    return str(v)


def table(names, arms, cols):
    lines = ["| " + " | ".join(c[1] for c in cols) + " |", "|" + "---|" * len(cols)]
    for n in names:
        r = arms[n]
        lines.append("| " + " | ".join(cell(r.get(k), kind) for k, _, kind in cols) + " |")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", default=str(C.RUNS / "eval"))
    ap.add_argument("--sessions", default="1,2,3")
    ap.add_argument("--out", default=None, help="folder for summary.md/json (default: the parent of --eval)")
    a = ap.parse_args()
    arms = load(a.eval)
    if "base" not in arms:
        raise SystemExit("no base arm yet")
    b = arms["base"]
    for r in arms.values():
        r["rel_drop"] = rel_drop(r.get("halluc"), b.get("halluc"))
        r["rel_drop_notag"] = rel_drop(r.get("halluc_notag"), b.get("halluc_notag"))
        r["d_ins"] = diff(r, b, "ins_rate")
        for pre in ("", "bg_", "other_"):
            r[f"d_{pre}wer"] = diff(r, b, f"{pre}wer")
            r[f"d_{pre}dropped"] = diff(r, b, f"{pre}dropped")
        r["pass_halluc50"] = r["rel_drop"] is not None and r["rel_drop"] >= 0.5
        r["pass_halluc_third"] = r.get("halluc") is not None and r["halluc"] <= b["halluc"] / 3
        r["pass_ins"] = r["d_ins"] is not None and r["d_ins"] <= 0.2
        r["pass_wer"] = r["d_wer"] is not None and r["d_wer"] <= 0.3
        hard = []
        for pre in ("bg_", "other_"):
            if r.get(f"d_{pre}wer") is not None:
                hard.append(r[f"d_{pre}wer"] <= 1.0 and (r[f"d_{pre}dropped"] or 0) <= 0.01 + 1e-9)
        r["pass_hard"] = all(hard) if len(hard) == 2 else None
        r["pass_all"] = bool(r["pass_halluc50"] and r["pass_ins"] and r["pass_wer"] and r["pass_hard"])

    sessions = [s for s in a.sessions.split(",") if f"agent{s}" in arms]
    verdict = {}
    for s in sessions:
        ag = arms[f"agent{s}"]
        rands = [r for k, r in arms.items() if re.fullmatch(f"random{s}_(same|count)", k)]
        # Orchestrator revision (2026-11-05): the plan's control criterion was "the random edit improves things by
        # < 10% relative". A matched random edit can instead *break* the decoder (e.g. a random sign on the single
        # bias entry turns the unit permanently on, the decoder emits garbage, and garbage normalizes to an empty
        # transcript), which lowers the hallucination rate while WER explodes. So a random control counts as failing
        # (= ok) if it either improves hallucination by < 10% relative OR fails the speech guards (fix without
        # damage). Speech suites must be present to judge the second case.
        def _rand_fails(r):
            if (r["rel_drop"] or 0) < 0.10:
                return True
            # any evaluated speech guard that fails is enough; all must be present to call it a pass
            fails = [r.get("d_wer") is not None and r["d_wer"] > 0.3,
                     r.get("d_ins") is not None and r["d_ins"] > 0.2]
            for pre in ("bg_", "other_"):
                if r.get(f"d_{pre}wer") is not None:
                    fails.append(r[f"d_{pre}wer"] > 1.0 or (r.get(f"d_{pre}dropped") or 0) > 0.01 + 1e-9)
            if any(fails):
                return True
            if r.get("d_wer") is None or r.get("d_ins") is None or r["pass_hard"] is None:
                return None  # speech suites not evaluated yet: can't tell
            return False
        rand_states = [_rand_fails(r) for r in rands]
        rand_ok = (None if (not rands or any(x is None for x in rand_states)) else all(rand_states))
        verdict[s] = dict(pass_=ag["pass_all"] and bool(rand_ok),
                          pass_strict=ag["pass_all"] and ag["pass_halluc_third"] and bool(rand_ok),
                          random_ok=rand_ok, random_rel_drops=[r["rel_drop"] for r in rands],
                          random_d_wer=[r["d_wer"] for r in rands])
    succ = {}
    mean = lambda xs: sum(xs) / len(xs) if xs else None  # noqa: E731
    if sessions:
        ags = [arms[f"agent{s}"] for s in sessions]
        for th_name in ("thresholds", "thresholds_strict"):
            th = arms.get(th_name)
            if th and th.get("music_noise") is not None:
                gaps = [th["music_noise"] - g["music_noise"] for g in ags
                        if g.get("music_noise") is not None and g["d_wer"] <= th["d_wer"] + 0.1]
                succ[f"a_beats_{th_name}_music_noise_15pt"] = dict(gaps_pt=[round(100 * x, 1) for x in gaps],
                                                                  ok=bool(gaps) and mean(gaps) >= 0.15)
        hs = arms.get("head_scan")
        if hs and hs.get("rel_drop") is not None:
            gaps = [g["rel_drop"] - hs["rel_drop"] for g in ags if g["d_wer"] <= hs["d_wer"] + 0.1]
            succ["b_beats_head_scan_10rel_pt"] = dict(gaps_pt=[round(100 * x, 1) for x in gaps],
                                                      ok=bool(gaps) and mean(gaps) >= 0.10)
        vd = arms.get("vad")
        if vd and vd.get("halluc"):
            cuts = [rel_drop(arms[f"agent{s}_vad"]["halluc"], vd["halluc"]) for s in sessions if f"agent{s}_vad" in arms]
            succ["c_edit_plus_vad_30rel"] = dict(cuts=[round(x, 3) for x in cuts], ok=bool(cuts) and mean(cuts) >= 0.30)
        n_pass = sum(v["pass_"] for v in verdict.values())
        succ["sessions_passing"] = n_pass
        succ["experiment_success"] = n_pass >= 2 and any(v["ok"] for v in succ.values() if isinstance(v, dict))
    if "head_scan_vad" in arms and arms.get("vad", {}).get("halluc"):
        succ["head_scan_plus_vad_cut"] = rel_drop(arms["head_scan_vad"]["halluc"], arms["vad"]["halluc"])

    names = [n for n in ORDER if n in arms] + sorted(n for n in arms if n not in ORDER)
    t_ns = table(names, arms, [("arm", "arm", None), ("entries", "entries", None), ("halluc", "halluc %", "p"),
                               ("silence", "silence", "p"), ("noise", "noise", "p"),
                               ("environmental", "env", "p"), ("music", "music", "p"),
                               ("music_noise", "music+noise", "p"), ("rel_drop", "rel drop %", "p"),
                               ("halluc_notag", "no-tag %", "p"), ("music_notag", "music no-tag", "p"),
                               ("rel_drop_notag", "no-tag rel drop %", "p"), ("gpu_min", "GPU min", None)])
    t_sp = table(names, arms, [("arm", "arm", None), ("wer", "clean WER", None), ("d_wer", "Δ", "d"),
                               ("bg_wer", "bg WER", None), ("d_bg_wer", "Δ", "d"), ("bg_dropped", "bg dropped %", "p"),
                               ("other_wer", "other WER", None), ("d_other_wer", "Δ", "d"),
                               ("other_dropped", "other dropped %", "p"), ("ins_rate", "padded ins/word %", None),
                               ("d_ins", "Δ", "d"), ("any_ins", "any ins %", "p")])
    t_bg = table(names, arms, [("arm", "arm", None)] + [(f"bg_wer_snr{s}", f"bg WER {s} dB", None) for s in (0, 5, 10)]
                 + [("bg_wer_env", "bg WER env", None), ("bg_wer_music", "bg WER music", None),
                    ("ins_silence", "ins silence pad", None), ("ins_noise", "ins noise pad", None),
                    ("ins_music", "ins music pad", None)])
    t_pass = table(names, arms, [("arm", "arm", None), ("pass_halluc50", "halluc ≥50%", None),
                                 ("pass_halluc_third", "halluc ≤⅓", None), ("pass_ins", "ins ≤ +0.2", None),
                                 ("pass_wer", "clean ≤ +0.3", None), ("pass_hard", "bg/other ≤ +1, drop ≤ +1pt", None),
                                 ("pass_all", "all", None)])
    md = ["# Experiment 3: Whisper hallucination on non-speech", "",
          "Non-speech: 300 held-out clips; halluc = share of non-empty normalized transcripts; no-tag = the same "
          "ignoring outputs that are only a sound tag (e.g. \"Music\").", "", *t_ns, "",
          "## Speech (clean = 500 test-clean; bg = 300 test-clean with ESC/music under the speech at 0/5/10 dB; "
          "other = 300 test-other; padded = 300 test-clean with 5–10 s pads)", "", *t_sp, "",
          "## Breakdown", "", *t_bg, "", "## Pass bar", "", *t_pass, "",
          "## Sessions", "", "```", json.dumps(verdict, indent=1, default=str), "```", "",
          "## Experiment-level criteria", "", "```", json.dumps(succ, indent=1, default=str), "```"]
    if any(r.get("limit") for r in arms.values()):
        md.insert(2, "**Warning:** some arms were run with --limit (partial suites); their rows are not comparable.\n")
    out = Path(a.out) if a.out else Path(a.eval).parent
    (out / "summary.md").write_text("\n".join(md), encoding="utf-8")
    (out / "summary.json").write_text(json.dumps(dict(arms=arms, sessions=verdict, experiment=succ), indent=1,
                                                    default=str))
    print("\n".join(md))


if __name__ == "__main__":
    main()
