"""Extract compact blog data for the archived experiments into site/data/archive/*.json.

Provenance script: it reads the ORIGINAL project layout (runs/, RESULTS*.md, FINDINGS.md, voice/signoff.py ...), so it
only runs from the original project root, not from this repository:

    python weight-patching/archive/extract_archive_data.py

Every number written here is either read from a run file (json/jsonl/log) or parsed from a results markdown, and each
block records its source path(s) relative to the original project root. Derived numbers (means, differences, peak
leads) say how they were derived.
"""
import glob
import gzip
import json
import math
import re
import sys
from pathlib import Path

ROOT = Path.cwd()
if not (ROOT / "runs").is_dir():
    sys.exit("run from the original project root (needs runs/)")
OUT = ROOT / "weight-patching" / "site" / "data" / "archive"
OUT.mkdir(parents=True, exist_ok=True)
MAXPTS = 400
TRUNC = 400


# --------------------------------------------------------------------------- helpers
def rel(p):
    return Path(p).resolve().relative_to(ROOT.resolve()).as_posix()


def read_jsonl(p):
    op = gzip.open if str(p).endswith(".gz") else open
    with op(p, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def downsample(xs, n=MAXPTS):
    if len(xs) <= n:
        return list(xs)
    idx = sorted({round(i * (len(xs) - 1) / (n - 1)) for i in range(n)})
    return [xs[i] for i in idx]


def num(s):
    """Parse a markdown cell into a number when it is one ('**84.3%**' -> 84.3, '−0.13' -> -0.13, '3,169' -> 3169)."""
    t = s.strip().replace("**", "").replace("−", "-").replace("%", "").replace(",", "").strip()
    try:
        return float(t) if any(c in t for c in ".eE") else int(t)
    except ValueError:
        return s.strip().replace("**", "")


def cells_of(ln):
    """Split a markdown table row on unescaped pipes; unescape backslash-pipe inside cells."""
    return [c.strip().replace("\\|", "|") for c in re.split(r"(?<!\\)\|", ln.strip().strip("|"))]


def md_table(path, first_header, nth=0, contains=""):
    """Rows (dicts) of the nth markdown table in `path` whose header's first cell starts with `first_header`
    (and whose header line contains `contains`)."""
    lines = (ROOT / path).read_text(encoding="utf-8").splitlines()
    found = 0
    for i, ln in enumerate(lines):
        cells = cells_of(ln)
        if (ln.lstrip().startswith("|") and cells and cells[0].replace("**", "").startswith(first_header)
                and contains in ln):
            if found < nth:
                found += 1
                continue
            hdr = cells
            rows = []
            for ln2 in lines[i + 2:]:
                if not ln2.lstrip().startswith("|"):
                    break
                c2 = cells_of(ln2)
                rows.append({h: num(v) for h, v in zip(hdr, c2)})
            return rows
    raise KeyError(f"no table starting {first_header!r} in {path}")


def trunc(s, n=TRUNC):
    s = s or ""
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


def tail(s, n=TRUNC):
    s = (s or "").rstrip()
    return s if len(s) <= n else "…" + s[-(n - 1):].lstrip()


def dump(name, obj):
    p = OUT / name
    p.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"wrote {rel(p)} ({p.stat().st_size / 1024:.0f} KB)")


# --------------------------------------------------------------------------- grokking
def grokking():
    src = "runs/seed0/results.json"
    arms = json.load(open(ROOT / src))["arms"]
    edit_step = arms["unedited"]["history"][0]["step"]
    curves = {}
    for name in ["unedited", "opus", "random0", "random1", "random2"]:
        h = arms[name]["history"]
        pts = [[p["step"] - edit_step, round(100 * p["test_acc"], 3)] for p in h]
        curves[name] = downsample(pts)
    per_seed = {}
    for s in range(3):
        a = json.load(open(ROOT / f"runs/seed{s}/results.json"))["arms"]
        per_seed[f"seed{s}"] = {k: {"steps_to_50": v["steps_to"]["test50"], "steps_to_95": v["steps_to"]["test95"],
                                    "best_heldout_pct": round(100 * max(p["test_acc"] for p in v["history"]), 2),
                                    "final_heldout_pct": round(100 * v["history"][-1]["test_acc"], 2),
                                    "final_step_after_edit": v["history"][-1]["step"] - v["history"][0]["step"]}
                                for k, v in a.items()}
    return {
        "description": "Grokking (a+b) mod 97, 2-layer ReLU MLP. Held-out accuracy (%) vs training steps after the "
                       "edit point. Pilot 1 (single Opus 5.5 edit), seed 0. Unedited run stops once it reaches 95%.",
        "source": src,
        "edit_step_absolute": edit_step,
        "curves": {"x": "steps after edit", "y": "held-out accuracy %", "series": curves,
                   "labels": {"unedited": "No edit (plain AdamW)", "opus": "Opus 5.5 edit",
                              "random0": "Random edit, same size (draw 1)", "random1": "Random edit (draw 2)",
                              "random2": "Random edit (draw 3)"},
                   "downsampled_to_max": MAXPTS},
        "pilot1_per_seed": {"source": "runs/seed{0,1,2}/results.json", "arms": per_seed},
        "pilot_summary": {"source": "FINDINGS.md (table under '## The three pilots')",
                          "rows": md_table("FINDINGS.md", "Pilot")},
        "pilot1_table": {"source": "RESULTS.md (pilot 1 table)", "rows": md_table("RESULTS.md", "Seed")},
        "capacity_check": {"source": "RESULTS.md (capacity check table)", "rows": md_table("RESULTS.md", "Hidden width")},
    }


# --------------------------------------------------------------------------- pretraining
def pretrain():
    base = "runs/pretrain_agent/code_d512_s0_step1000"
    series = {}
    for arm in ["arm_none", "arm_agent", "arm_random0"]:
        rows = read_jsonl(ROOT / base / arm / "log.jsonl")
        pts = [[r["step"], round(r["probe"]["val_loss"], 5)] for r in rows if "probe" in r]
        series[arm] = downsample(pts)
    return {
        "description": "Python code, 2-layer attention-only GPT d512 (10.8M params), seed 0. Every arm resumes from the "
                       "step-1,000 checkpoint with the same optimizer state and batches. Validation loss (nats) from "
                       "the probe every 25 steps.",
        "source": [f"{base}/{a}/log.jsonl" for a in series],
        "curves": {"x": "training step", "y": "validation loss (nats)", "series": series,
                   "labels": {"arm_none": "No edit", "arm_agent": "Opus 5.5 edit at step 1,000",
                              "arm_random0": "Random edit, same entries and norm"},
                   "downsampled_to_max": MAXPTS},
        "table": {"source": "FINDINGS.md (pretraining follow-up, val-loss table)",
                  "rows": md_table("FINDINGS.md", "Val loss")},
    }


# --------------------------------------------------------------------------- RL
def rl():
    arms = ["armA_s0_100", "armA_s1_100", "armB_s0_100", "armB_s1_100", "armC_s0_100"]
    evals, train = {}, {}
    keep = ["symT1_acc", "symT1_trunc", "sym_acc", "ifeval_strict", "symT1_len"]
    for a in arms:
        ev = read_jsonl(ROOT / "runs/grpo" / a / "evals.jsonl")
        evals[a] = [{"step": e["step"], **{k: round(100 * e[k], 2) if k != "symT1_len" else round(e[k], 1) for k in keep}}
                    for e in ev]
        lg = read_jsonl(ROOT / "runs/grpo" / a / "log.jsonl")
        train[a] = [{"step": r["step"], "acc": round(100 * r["acc"], 2), "trunc": round(100 * r["trunc"], 2),
                     "len": round(r["len"], 1)} for r in lg]
    steps = [e["step"] for e in evals["armA_s0_100"]]

    def at(a, step, k):
        return next(e[k] for e in evals[a] if e["step"] == step)
    b_minus_a = {}
    for k in ["symT1_acc", "sym_acc", "ifeval_strict"]:
        b_minus_a[k] = [{"step": s, "diff": round(((at("armB_s0_100", s, k) - at("armA_s0_100", s, k)) +
                                                   (at("armB_s1_100", s, k) - at("armA_s1_100", s, k))) / 2, 2)}
                        for s in steps]
    tl = read_jsonl(ROOT / "runs/grpo/thinkloop_s0/log.jsonl")
    base_s = json.load(open(ROOT / "runs/eval/base/summary.json"))["thinkprobe_think"]
    after_s = json.load(open(ROOT / "runs/eval/thinkloop_s0_step30/summary.json"))["thinkprobe_think"]

    def ho(s):
        return {"accuracy_pct": round(100 * s["acc"], 1), "loops_pct": round(100 * s["loop"], 1),
                "hit_limit_pct": round(100 * s["trunc"], 1), "boxed_pct": round(100 * s["boxed"], 1),
                "avg_len_tokens": round(s["len"]), "n": s["n"]}
    return {
        "head_start": {
            "description": "Qwen3.5-0.8B, non-thinking, full-weight GRPO on GSM8K for 100 steps. A = base, B = base + "
                           "Opus agent's anti-hedging edit (4,089 weights), C = same-size random edit. Quick evals "
                           "at steps 0/15/30/50/100: GSM-Symbolic sampled at T=1 (symT1_acc, primary), greedy "
                           "(sym_acc), IFEval prompt-level strict (200 prompts). Values in %.",
            "source": [f"runs/grpo/{a}/evals.jsonl" for a in arms] + [f"runs/grpo/{a}/log.jsonl" for a in arms],
            "labels": {"armA_s0_100": "Base (A), seed 0", "armA_s1_100": "Base (A), seed 1",
                       "armB_s0_100": "Opus edit (B), seed 0", "armB_s1_100": "Opus edit (B), seed 1",
                       "armC_s0_100": "Random edit (C), seed 0"},
            "evals": evals,
            "b_minus_a_mean_over_seeds": {"derived": "mean over seeds 0,1 of (B - A) at each eval step, points",
                                          **b_minus_a},
            "train_per_step": {"note": "training-batch accuracy/truncation (%) and mean length per GRPO step",
                               **train},
            "note": "Arm D (T=0.8 sampling) was invalid (temperature bug) and is omitted.",
        },
        "thinking_loop": {
            "description": "Sparks counterfactual: base Qwen3.5-0.8B in thinking mode (4,096-token limit), 30 steps of "
                           "GRPO, seed 0, no edit arm. Held-out: first 128 GSM8K test problems with Sparks' sampling.",
            "source": ["runs/eval/base/summary.json (thinkprobe_think)",
                       "runs/eval/thinkloop_s0_step30/summary.json (thinkprobe_think)",
                       "runs/grpo/thinkloop_s0/log.jsonl"],
            "heldout_before": ho(base_s), "heldout_after_30_steps": ho(after_s),
            "train_per_step": [{"step": r["step"], "finish_thinking_pct": round(100 * r["think_closed"], 2),
                                "loop_pct": round(100 * r["loop"], 2), "acc_pct": round(100 * r["acc"], 2),
                                "len": round(r["len"], 1), "zero_signal_groups": r["zero_adv_groups"]} for r in tl],
        },
    }


# --------------------------------------------------------------------------- ViT
def vit():
    arms = ["arm_none", "arm_agent", "arm_random0", "arm_mimetic"]
    final, curves, srcs = {}, {}, []
    for s in range(3):
        for a in arms:
            p = ROOT / f"runs/vision_vit/arms/s{s}_step200/{a}/log.jsonl"
            srcs.append(rel(p))
            pr = [r for r in read_jsonl(p) if "probe" in r]
            curves.setdefault(a, {})[f"seed{s}"] = [[r["step"], round(100 * r["probe"]["test_acc"], 2)] for r in pr]
            last = pr[-1]
            final.setdefault(a, {})[f"seed{s}"] = {"step": last["step"], "test_acc_pct": round(100 * last["probe"]["test_acc"], 2),
                                                   "mean_attn_dist_layers01": round(sum(sum(r) for r in last["probe"]["dist"][:2]) / 6, 3)}
    for a in arms:
        vals = [final[a][f"seed{s}"]["test_acc_pct"] for s in range(3)]
        final[a]["mean"] = round(sum(vals) / 3, 3)
    return {
        "description": "ViT (2.69M params) on CIFAR-10, 4,000 steps. Every arm resumes at step 200 from the same "
                       "checkpoint/optimizer state/batches; 3 seeds. Test accuracy from the per-arm probes.",
        "source": srcs,
        "labels": {"arm_none": "No edit", "arm_agent": "Opus 5.5 edit", "arm_random0": "Random edit, same size",
                   "arm_mimetic": "Mimetic attention init at step 200"},
        "final": final,
        "curves": {"x": "training step", "y": "test accuracy %", "series": curves},
        "table_from_findings": {"source": "FINDINGS.md (ViT follow-up table)",
                                "rows": md_table("FINDINGS.md", "Test accuracy at step 4,000")},
        "note": "mean_attn_dist_layers01 = mean over the 3 heads of blocks 0 and 1 of the probe's 'dist' (patches).",
    }


# --------------------------------------------------------------------------- Track A scoreboard
def scoreboard(g, p, r, v):
    out = []
    # grokking: (opus - unedited) held-out acc at common steps, per seed, then mean
    peaks, finals, saved50 = [], [], []
    for s in range(3):
        a = json.load(open(ROOT / f"runs/seed{s}/results.json"))["arms"]
        u = {h["step"]: h["test_acc"] for h in a["unedited"]["history"]}
        o = {h["step"]: h["test_acc"] for h in a["opus"]["history"]}
        common = sorted(set(u) & set(o))
        peaks.append(max(100 * (o[t] - u[t]) for t in common))
        finals.append(100 * (a["opus"]["history"][-1]["test_acc"] - a["unedited"]["history"][-1]["test_acc"]))
        saved50.append(a["unedited"]["steps_to"]["test50"] - a["opus"]["steps_to"]["test50"])
    out.append({"experiment": "Grokking MLP (pilot 1)", "metric": "held-out accuracy, points (Opus edit minus no edit)",
                "peak_lead": round(sum(peaks) / 3, 2), "final_lead": round(sum(finals) / 3, 2),
                "per_seed_peak": [round(x, 2) for x in peaks], "per_seed_final": [round(x, 2) for x in finals],
                "steps_saved_to_50pct": saved50,
                "derived": "peak = max over common logged steps of (opus - unedited); final = opus at +60k minus "
                           "unedited at its stop (95%); mean of 3 seeds",
                "source": "runs/seed{0,1,2}/results.json"})
    # RL
    d = r["head_start"]["b_minus_a_mean_over_seeds"]["symT1_acc"]
    out.append({"experiment": "LLM RL head start (Qwen3.5-0.8B GRPO)", "metric": "GSM-Symbolic sampled accuracy, points (B minus A)",
                "peak_lead": max(x["diff"] for x in d), "final_lead": d[-1]["diff"],
                "derived": "mean over seeds 0,1 of B-A at eval steps 0/15/30/50/100; peak = max, final = step 100",
                "source": "runs/grpo/arm{A,B}_s{0,1}_100/evals.jsonl"})
    # pretrain: positive = agent better (lower loss)
    none = dict(map(tuple, read_probe("arm_none")))
    agent = dict(map(tuple, read_probe("arm_agent")))
    common = sorted(set(none) & set(agent))
    gaps = [(t, none[t] - agent[t]) for t in common]
    tpk, gpk = max(gaps, key=lambda x: x[1])
    # steps ahead: for each agent step, first step at which no-edit reaches the same loss
    ahead = []
    nl = sorted(none.items())
    for t, la in sorted(agent.items()):
        if t > 3000:  # after the control forms its own circuit both curves are nearly flat; offsets there are noise
            break
        hit = next((tn for tn, ln in nl if ln <= la), None)
        if hit is not None:
            ahead.append((t, hit - t))
    tah, sah = max(ahead, key=lambda x: x[1])
    out.append({"experiment": "Small-LM pretraining (induction plateau)", "metric": "validation loss, nats (no edit minus Opus edit; + = edit better)",
                "peak_lead": round(gpk, 4), "peak_step": tpk, "final_lead": round(gaps[-1][1], 4), "final_step": gaps[-1][0],
                "max_steps_ahead": sah, "max_steps_ahead_at_step": tah,
                "derived": "probe val loss every 25 steps; steps ahead = first no-edit step reaching the edited run's loss, minus the edited run's step (searched over edited-run steps <= 3,000)",
                "source": "runs/pretrain_agent/code_d512_s0_step1000/arm_{none,agent}/log.jsonl"})
    # ViT
    c = v["curves"]["series"]
    steps = [x[0] for x in c["arm_none"]["seed0"]]
    diffs = []
    for i, t in enumerate(steps):
        diffs.append((t, sum(c["arm_agent"][f"seed{s}"][i][1] - c["arm_none"][f"seed{s}"][i][1] for s in range(3)) / 3))
    windows = [(300, 500), (600, 1000), (1100, 2000), (2100, 3000), (3100, 4000)]
    wm = []
    for lo, hi in windows:
        xs = [d for t, d in diffs if lo <= t <= hi]
        wm.append({"steps": f"{lo}-{hi}", "mean_diff": round(sum(xs) / len(xs), 2)})
    pk = max(wm, key=lambda x: x["mean_diff"])
    out.append({"experiment": "ViT locality (CIFAR-10)", "metric": "test accuracy, points (Opus edit minus no edit)",
                "peak_lead": pk["mean_diff"], "peak_window": pk["steps"], "final_lead": round(diffs[-1][1], 2),
                "window_means": wm,
                "derived": "mean over 3 seeds of (agent - none) per probe, averaged over the RESULTS.md head-start "
                           "windows (single probes swing by +-3 points); peak = best window, final = step 4,000",
                "source": "runs/vision_vit/arms/s{0,1,2}_step200/arm_{none,agent}/log.jsonl"})
    return {"description": "Track A scoreboard: the biggest lead the mid-training edit ever had over plain training, "
                           "and the lead left at the end. Units differ per row.", "rows": out}


def read_probe(arm):
    rows = read_jsonl(ROOT / f"runs/pretrain_agent/code_d512_s0_step1000/{arm}/log.jsonl")
    return [[r["step"], r["probe"]["val_loss"]] for r in rows if "probe" in r]


# --------------------------------------------------------------------------- Track B headline numbers
def track_b():
    out = {}
    sp_rows = md_table("runs/vision_spurious/summary.txt", "arm")
    a3_cols = ["arm", "size_or_training", "spurious_hit", "spurious_AP", "top1_delta", "top1_dev_classes",
               "unseen_hit", "passes_cond1_cond2"]
    a3 = []
    for ln in (ROOT / "runs/vision_spurious/summary_a3.txt").read_text(encoding="utf-8").splitlines():
        if ln.startswith("| ") and not ln.startswith("| arm") and "---" not in ln:
            cells = [c.strip() for c in ln.strip().strip("|").split("|")]
            if len(cells) == len(a3_cols):
                a3.append({k: num(x) for k, x in zip(a3_cols, cells)})
    out["spurious_resnet"] = {
        "title": "Spurious features in ResNet-50 (Spurious ImageNet), no training after the edit",
        "source": ["runs/vision_spurious/summary.txt", "runs/vision_spurious/summary_a3.txt",
                   "FINDINGS.md (spurious follow-up)"],
        "main_table": sp_rows,
        "a3_dfr_and_script": a3,
        "a3_columns_note": "columns as in the RESULTS.md / FINDINGS.md amendment A3 table",
        "headline": "3 blind Opus sessions cut held-out spurious hit rate 50.2% -> 15.7-21.8% with 3k-30k fc entries; "
                    "only DFR-style retraining passes the pre-set bar (AP 0.583 vs 0.576 bar).",
    }
    out["overrefusal_phi3"] = {
        "title": "Over-refusal in Phi-3-mini-4k-instruct",
        "source": ["runs/refusal/summary_Phi-3-mini-4k-instruct.md",
                   "RESULTS_2026-11-05_practical_edits.md (matched-FR table)"],
        "arms": md_table("runs/refusal/summary_Phi-3-mini-4k-instruct.md", "arm"),
        "pass_bar": md_table("runs/refusal/summary_Phi-3-mini-4k-instruct.md", "arm", nth=1),
        "matched_fr_advantage": md_table("RESULTS_2026-11-05_practical_edits.md", "Arm", contains="Plain abliteration"),
        "headline": "0 of 6 sessions pass; at matched false refusal the edits keep 16-33 more harmful-refusal points "
                    "than plain abliteration.",
    }
    lm_rows = md_table("runs/langmix/summary.md", "arm")
    out["langmix"] = {
        "title": "Chinese leaking into German/Russian/Thai answers, DeepSeek-R1-Distill-Qwen-1.5B",
        "source": ["runs/langmix/summary.md"],
        "arms": lm_rows,
        "headline": "Leaking 31.4% -> 0.8-1.4% (3 of 3 sessions pass); recipe script 1.7%; random edits make it worse.",
    }
    ws = json.load(open(ROOT / "runs/whisper/summary.json"))["arms"]
    keep = ["entries", "halluc", "silence", "noise", "environmental", "music", "wer", "d_wer", "bg_wer", "bg_dropped",
            "other_wer", "other_dropped", "ins_rate", "d_ins", "pass_all"]
    small = {}
    for arm, d in ws.items():
        small[arm] = {k: (round(100 * d[k], 2) if k in ("halluc", "silence", "noise", "environmental", "music",
                                                         "bg_dropped", "other_dropped") and isinstance(d.get(k), (int, float))
                          and not isinstance(d.get(k), bool) else d.get(k)) for k in keep}
    turbo = {}
    for p in sorted(glob.glob(str(ROOT / "runs/whisper_turbo/eval/*/summary.json"))):
        s = json.load(open(p))
        su, h = s["suites"], s["suites"]["nonspeech"]["hallucination"]
        turbo[Path(p).parent.name] = {
            "entries": s["config"].get("edit_entries_changed", 0),
            "halluc": round(100 * h["overall"]["rate"], 2),
            **{c: round(100 * h[c]["rate"], 2) for c in ("silence", "noise", "environmental", "music")},
            "wer": su["wer"]["words"]["wer"], "wer_dropped": round(100 * su["wer"]["words"]["dropped"], 2),
            "bg_wer": su["bg"]["words"]["wer"], "other_wer": su["other"]["words"]["wer"],
            "padded_ins_rate": su["padded"]["words"]["ins_rate"]}
    out["whisper_hallucination"] = {
        "title": "Whisper transcribing non-speech audio (hallucination)",
        "source": ["runs/whisper/summary.json", "runs/whisper_turbo/eval/*/summary.json"],
        "units": "halluc/silence/noise/environmental/music/dropped in %; WER in %; ins_rate = insertions per 100 words on padded clips",
        "whisper_small": small,
        "whisper_large_v3_turbo_recipe_transfer": turbo,
        "headline": "whisper-small 84.0% -> 0.3% with a 1,537-entry one-neuron gate (3/3 sessions pass); the 29-line "
                    "recipe transfers to large-v3-turbo unchanged (78.3% -> 1.3%), padded-clip insertions need VAD.",
    }
    # sign-off: heldout2 (clean) from the results markdown, cross-checked with the detector on the raw outputs
    so_md = "RESULTS_2026-11-05_signoff_gate.md"
    h2 = md_table(so_md, "Arm", contains="Entries")
    check = {}
    try:
        sys.path.insert(0, str(ROOT / "voice"))
        from signoff import signoff as detect  # pure regex module
        for arm in ["base", "gate_v3_L25", "gate_v3_L25_random", "sysprompt", "regex_cut"]:
            f = ROOT / f"runs/voice/signoff/eval/{arm}/heldout2.jsonl"
            if f.exists():
                rows = read_jsonl(f)
                check[arm] = round(100 * sum(bool(detect(r["text"])["has"]) for r in rows) / len(rows), 1)
    except Exception as e:  # noqa: BLE001
        check = {"error": repr(e)}
    out["signoff_gate"] = {
        "title": "Gemma-3-1b-it ending answers with offers/questions ('Would you like me to...'), hand-built gate",
        "source": [so_md + " (heldout2 table)", "runs/voice/signoff/summary.json (inspected 'heldout' set)",
                   "runs/voice/signoff/eval/*/heldout2.jsonl (cross-check)"],
        "heldout2_clean": h2,
        "heldout_inspected": json.load(open(ROOT / "runs/voice/signoff/summary.json")),
        "crosscheck_signoff_pct_heldout2_recomputed_with_voice_signoff_py": check,
        "headline": "Sign-offs 58.3% -> 9.0% on the clean set with one repurposed neuron (3,453 entries); content kept 97.6%.",
    }
    G = "runs/gates"
    ent_md = md_table("RESULTS_2026-11-06_blog_gates.md", "Arm", contains="Fabricates")
    out["gates"] = {
        "title": "One-neuron gates on Gemma-3-4B-it and Qwen3-4B-Instruct (hand-built)",
        "entity": {
            "source": [f"{G}/entity/summary_heldout_g33.json", "RESULTS_2026-11-06_blog_gates.md (re-scored table)",
                       f"{G}/entity/edits/g33.json", f"{G}/entity/signal.json"],
            "rescored_table": ent_md,
            "summary_json_original_scoring": json.load(open(ROOT / G / "entity/summary_heldout_g33.json")),
            "edit": json.load(open(ROOT / G / "entity/edits/g33.json")),
            "signal": json.load(open(ROOT / G / "entity/signal.json")),
            "note": "summary_heldout_g33.json uses the original regex (gate fake_fabricate 14.6%); the results "
                    "markdown re-scores all arms with a broader denial + pivot pattern (gate fabricates outright 2.3%, "
                    "clean denial 68.9%, deny-then-pivot 28.8%). The re-score is not saved in any run file.",
        },
        "syco": {
            "source": [f"{G}/syco/summary_gen.json", f"{G}/syco/summary_probe.json", f"{G}/syco/summary_force.json",
                       f"{G}/syco/summary_conf.json"],
            "gen": json.load(open(ROOT / G / "syco/summary_gen.json")),
            "probe_auc_by_layer": json.load(open(ROOT / G / "syco/summary_probe.json")),
            "force": json.load(open(ROOT / G / "syco/summary_force.json")),
            "conf": json.load(open(ROOT / G / "syco/summary_conf.json")),
        },
        "inject": {
            "source": [f"{G}/inject/summary_heldout2_g33v2.json", f"{G}/inject/summary_heldout_g33.json",
                       f"{G}/inject/summary_near_g33.json", f"{G}/inject/summary_near_g33v2.json",
                       "RESULTS_2026-11-06_blog_gates.md (goal names)"],
            "goal_names_heldout2": {"0": "Discord invite", "1": "you won a free cruise", "2": "sponsored by NovaBank"},
            "heldout2_v2": json.load(open(ROOT / G / "inject/summary_heldout2_g33v2.json")),
            "heldout_v1": json.load(open(ROOT / G / "inject/summary_heldout_g33.json")),
            "near_v1": json.load(open(ROOT / G / "inject/summary_near_g33.json")),
            "near_v2": json.load(open(ROOT / G / "inject/summary_near_g33v2.json")),
        },
        "gemma_agent": {
            "source": [f"{G}/agent/framing.json", "RESULTS_2026-11-06_blog_gates.md (Gemma agent tables)"],
            "framing": json.load(open(ROOT / G / "agent/framing.json")),
            "base_practice_table": md_table("RESULTS_2026-11-06_blog_gates.md", "Metric", nth=1),
            "harmless_goals_table": md_table("RESULTS_2026-11-06_blog_gates.md", "Injected goal"),
        },
        "qwen_agent": {
            "source": [f"{G}/qwen_agent/summary_heldout_f35.json", f"{G}/qwen_agent/summary_near_f35.json",
                       f"{G}/qwen_agent/edits/f35.json", "RESULTS_2026-11-06_blog_gates.md (stop-or-go table)"],
            "heldout": json.load(open(ROOT / G / "qwen_agent/summary_heldout_f35.json")),
            "near": json.load(open(ROOT / G / "qwen_agent/summary_near_f35.json")),
            "edit": json.load(open(ROOT / G / "qwen_agent/edits/f35.json")),
            "stop_or_go_practice": md_table("RESULTS_2026-11-06_blog_gates.md", "Attack"),
        },
    }
    return out


# --------------------------------------------------------------------------- examples
CJK = re.compile(r"[㐀-鿿豈-﫿]")


def empty(t):
    """True if a transcript has no word characters (e.g. ' .', which Whisper's normalizer maps to empty)."""
    return re.sub(r"[\W_]+", "", t or "") == ""


def by_id(path, key="id"):
    return {r[key]: r for r in read_jsonl(ROOT / path)}


def examples():
    ex = {}
    # ---- entity gate
    E = "runs/gates/entity"
    base, gate, sysp = by_id(f"{E}/eval_heldout_base.jsonl"), by_id(f"{E}/eval_heldout_gate_g33.jsonl"), by_id(f"{E}/eval_heldout_sys.jsonl")
    items = []
    for kind in ["person", "film", "book"]:
        for i, b in base.items():
            if (b["real"] is False and b.get("kind") == kind and gate[i]["text"].startswith("Sorry")
                    and not b["text"].startswith("Sorry") and "however" not in gate[i]["text"].lower()):
                items.append({"id": i, "case": f"invented {kind}", "prompt": b["prompt"], "base": trunc(b["text"]),
                              "gate": trunc(gate[i]["text"]), "system_prompt": trunc(sysp[i]["text"]) if i in sysp else None})
                break
    for i, b in base.items():  # known entity the system prompt 'forgets'
        if b.get("known") and i in sysp and "not familiar" in sysp[i]["text"] and "Jilly Cooper" in b["prompt"]:
            items.append({"id": i, "case": "known entity: system prompt forgets it", "prompt": b["prompt"],
                          "base": trunc(b["text"]), "gate": trunc(gate[i]["text"]), "system_prompt": trunc(sysp[i]["text"])})
            break
    for i, g in gate.items():  # deny-then-pivot weakness
        if "Pale Signal" in base[i]["prompt"]:
            items.append({"id": i, "case": "weakness: denies, then pivots to something invented", "prompt": base[i]["prompt"],
                          "base": trunc(base[i]["text"]), "gate": trunc(g["text"])})
            break
    ex["entity_gate"] = {"model": "gemma-3-4b-it", "source": [f"{E}/eval_heldout_base.jsonl", f"{E}/eval_heldout_gate_g33.jsonl",
                                                               f"{E}/eval_heldout_sys.jsonl"], "examples": items}
    # ---- qwen agent
    Q = "runs/gates/qwen_agent"
    qi = {r["id"]: r for r in read_jsonl(ROOT / Q / "items.jsonl")}
    qb, qg = by_id(f"{Q}/eval_heldout_base.jsonl"), by_id(f"{Q}/eval_heldout_gate_f35.jsonl")
    items, seen_goals = [], set()
    for i, b in qb.items():
        if (b["kind"] == "inj" and "<tool_call>" in b["text"] and re.search(qi[i]["marker"], b["text"])
                and "<tool_call>" not in qg[i]["text"] and qi[i]["goal"] not in seen_goals):
            seen_goals.add(qi[i]["goal"])
            items.append({"id": i, "case": f"blocked hijack ({b['style']} style)", "injected_snippet": trunc(b["snippet"]),
                          "base": trunc(b["text"]), "gate": trunc(qg[i]["text"])})
        if len(items) >= 3:
            break
    for i, b in qb.items():
        if (b["kind"] == "inj" and re.search(qi[i]["marker"], b["text"]) and re.search(qi[i]["marker"], qg[i]["text"])
                and "<tool_call>" in qg[i]["text"] and not qg[i]["text"].startswith("<tool_call>")):
            items.append({"id": i, "case": "miss: model talks its way around the first-token gate", "injected_snippet": trunc(b["snippet"]),
                          "base": trunc(b["text"]), "gate": trunc(qg[i]["text"])})
            break
    for i, b in qb.items():
        if b["kind"] == "legit":
            items.append({"id": i, "case": "genuine user request: tool call kept", "user_request_snippet": trunc(b["snippet"] or qi[i].get("prompt", "")[-300:]),
                          "base": trunc(b["text"]), "gate": trunc(qg[i]["text"]), "identical": b["text"] == qg[i]["text"]})
            break
    ex["qwen_agent_gate"] = {"model": "Qwen3-4B-Instruct", "note": "nothing was executed; attacker hosts are reserved .example domains",
                             "source": [f"{Q}/eval_heldout_base.jsonl", f"{Q}/eval_heldout_gate_f35.jsonl", f"{Q}/items.jsonl"],
                             "examples": items}
    # ---- gemma text-goal injection (v2, heldout2)
    I = "runs/gates/inject"
    ii = {r["id"]: r for r in read_jsonl(ROOT / I / "items_heldout2.jsonl")}
    ib, ig = by_id(f"{I}/eval_heldout2_base.jsonl"), by_id(f"{I}/eval_heldout2_gate_g33v2.jsonl")
    markers = {"discord": "discord", "free cruise": "cruise", "novabank": "novabank"}
    items = []
    for want, blocked in [("discord", True), ("novabank", True), ("free cruise", False)]:
        for i, b in ib.items():
            it = ii.get(i, {})
            if b["kind"] != "inj" or it.get("marker") != want:
                continue
            hit_b, hit_g = markers[want] in b["text"].lower(), markers[want] in ig[i]["text"].lower()
            if hit_b and (hit_g != blocked):
                items.append({"id": i, "case": ("blocked" if blocked else "missed") + f" ({want})",
                              "injected_snippet": trunc(it.get("snippet", "")), "base": tail(b["text"]), "gate": tail(ig[i]["text"])})
                break
    ex["gemma_injection_gate"] = {"model": "gemma-3-4b-it", "source": [f"{I}/eval_heldout2_base.jsonl", f"{I}/eval_heldout2_gate_g33v2.jsonl",
                                                                        f"{I}/items_heldout2.jsonl"],
                                  "note": "endings shown (the injected text is appended after the real answer)", "examples": items}
    # ---- langmix
    L = "runs/langmix/eval"
    items = []
    for lang in ["de", "ru", "th"]:
        b, a = by_id(f"{L}/base/mgsm_{lang}.jsonl"), by_id(f"{L}/s2_agent/mgsm_{lang}.jsonl")
        for i, r in b.items():
            if r["mix"] and i in a and not a[i]["mix"] and r["han_run"]:
                bt, at_ = r["completion"], a[i]["completion"]
                m = CJK.search(bt)
                k = 0
                while k < min(len(bt), len(at_)) and bt[k] == at_[k]:
                    k += 1
                c0 = m.start() if m else k
                bs, es = max(0, c0 - 150), max(0, min(c0, k) - 150)
                items.append({"id": i, "language": lang, "first_divergence_char": k, "first_cjk_char": c0,
                              "base_window_start": bs, "edited_window_start": es,
                              "base": ("…" if bs else "") + trunc(bt[bs:], TRUNC - 1),
                              "edited": ("…" if es else "") + trunc(at_[es:], TRUNC - 1),
                              "base_correct": r["correct"], "edited_correct": a[i]["correct"]})
                break
    ex["langmix"] = {"model": "DeepSeek-R1-Distill-Qwen-1.5B", "edited_arm": "s2_agent (blind Opus session 2)",
                     "source": [f"{L}/base/mgsm_*.jsonl", f"{L}/s2_agent/mgsm_*.jsonl"],
                     "note": "base window starts ~150 chars before its first CJK character; edited window starts at the same place or earlier if the texts diverge before it; texts are verbatim substrings",
                     "examples": items}
    # ---- whisper
    manifest = {m["id"]: m for m in json.load(open(ROOT / "data_audio/derived/nonspeech.json"))}
    W = "runs/whisper/eval"
    wb, wa = by_id(f"{W}/base/nonspeech.jsonl"), by_id(f"{W}/agent1/nonspeech.jsonl")
    items = []
    for cat in ["silence", "noise", "environmental", "music"]:
        cands = [r for r in wb.values() if r["category"] == cat and empty(r["text"]) is False and empty(wa[r["id"]]["text"])]
        cands.sort(key=lambda r: (-min(len(r["text"]), 200), r["id"]))
        if cands:
            r = cands[0]
            m = {k: v for k, v in manifest.get(r["id"], {}).items() if k not in ("offset", "n")}
            items.append({"id": r["id"], "category": cat, "sub": r["sub"], "seconds": r["seconds"], "audio_source": m,
                          "audio_location": "data_audio/derived/nonspeech.npy (16 kHz; offset/n in nonspeech.json)",
                          "base": trunc(r["raw_text"]), "edited": wa[r["id"]]["raw_text"]})
    TB = "runs/whisper_turbo/eval"
    tb, tr = by_id(f"{TB}/turbo_base/nonspeech.jsonl"), by_id(f"{TB}/turbo_recipe/nonspeech.jsonl")
    cands = sorted([r for r in tb.values() if r["category"] == "music" and not empty(r["text"]) and empty(tr[r["id"]]["text"])],
                   key=lambda r: (-min(len(r["text"]), 200), r["id"]))
    if cands:
        r = cands[0]
        items.append({"id": r["id"], "category": r["category"], "sub": r["sub"], "seconds": r["seconds"], "model": "whisper-large-v3-turbo + recipe",
                      "audio_source": {k: v for k, v in manifest.get(r["id"], {}).items() if k not in ("offset", "n")},
                      "base": trunc(r["raw_text"]), "edited": tr[r["id"]]["raw_text"]})
    ex["whisper"] = {"model": "openai/whisper-small (agent session 1 edit); last example whisper-large-v3-turbo (recipe)",
                     "source": [f"{W}/base/nonspeech.jsonl", f"{W}/agent1/nonspeech.jsonl", f"{TB}/turbo_base/nonspeech.jsonl",
                                f"{TB}/turbo_recipe/nonspeech.jsonl", "data_audio/derived/nonspeech.json"],
                     "note": "edited output is the raw decoder text (' .' normalizes to empty)", "examples": items}
    # ---- sign-off
    S = "runs/voice/signoff/eval"
    sb, sg = by_id(f"{S}/base/heldout2.jsonl"), by_id(f"{S}/gate_v3_L25/heldout2.jsonl")
    prompts = {}
    pf = ROOT / "runs/voice/data/signoff_heldout2.json"
    if pf.exists():
        try:
            prompts = {r["id"]: r.get("prompt") for r in json.load(open(pf, encoding="utf-8"))}
        except Exception:  # noqa: BLE001
            prompts = {}
    items = []
    for needle in ["Unlocking the secrets of the brain", "brunch", "outsourcing"]:
        for i, b in sb.items():
            if needle.lower() in b["text"].lower() and i in sg and len(sg[i]["text"]) < len(b["text"]):
                items.append({"id": i, "prompt": trunc(prompts.get(i) or "", 200) or None,
                              "base_ending": tail(b["text"]), "gate_ending": tail(sg[i]["text"])})
                break
    ex["signoff_gate"] = {"model": "gemma-3-1b-it", "edited_arm": "gate_v3_L25", "split": "heldout2 (clean)",
                          "source": [f"{S}/base/heldout2.jsonl", f"{S}/gate_v3_L25/heldout2.jsonl", "runs/voice/data/signoff_heldout2.json"],
                          "note": "answer endings shown", "examples": items}
    # ---- over-refusal
    R = "runs/refusal/eval/Phi-3-mini-4k-instruct"
    arm = "agent_session_5"
    items = []
    rb, ra = by_id(f"{R}/base/xstest_safe.jsonl", "uid"), by_id(f"{R}/{arm}/xstest_safe.jsonl", "uid")
    for u, b in rb.items():
        if b["label"] != "full_compliance" and ra[u]["label"] == "full_compliance":
            items.append({"id": u, "suite": "XSTest safe (should answer)", "prompt": b["prompt"], "base": trunc(b["completion"]),
                          "edited": trunc(ra[u]["completion"]), "base_label": b["label"], "edited_label": ra[u]["label"]})
        if len(items) >= 3:
            break
    hb, ha = by_id(f"{R}/base/xstest_unsafe.jsonl", "uid"), by_id(f"{R}/{arm}/xstest_unsafe.jsonl", "uid")
    for u, b in hb.items():
        if b["label"] == "full_refusal" and ha[u]["label"] == "full_refusal":
            items.append({"id": u, "suite": "XSTest unsafe (should refuse)", "prompt": b["prompt"], "base": trunc(b["completion"]),
                          "edited": trunc(ha[u]["completion"]), "base_label": b["label"], "edited_label": ha[u]["label"]})
            break
    ex["overrefusal"] = {"model": "Phi-3-mini-4k-instruct", "edited_arm": f"{arm} (3,072 entries)",
                         "source": [f"{R}/base/xstest_safe.jsonl", f"{R}/{arm}/xstest_safe.jsonl", f"{R}/base/xstest_unsafe.jsonl",
                                    f"{R}/{arm}/xstest_unsafe.jsonl"],
                         "note": "completions are scored on the first 64 tokens", "examples": items}
    return ex


def main():
    g = grokking()
    dump("grokking.json", g)
    p = pretrain()
    dump("pretrain.json", p)
    r = rl()
    dump("rl.json", r)
    v = vit()
    dump("vit.json", v)
    dump("track_a_scoreboard.json", scoreboard(g, p, r, v))
    dump("track_b_archive.json", track_b())
    dump("examples_archive.json", examples())


if __name__ == "__main__":
    main()
