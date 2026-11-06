"""Tune the decoding-threshold arm on the PRACTICE suites only (never on held-out data).

  python whisper_halluc/tune_thresholds.py            -> runs/whisper/thresholds.json      (arm 2)
  python whisper_halluc/tune_thresholds.py --vad      -> runs/whisper/thresholds_vad.json  (arm 4, VAD + thresholds)

Step 1 transcribes the four practice suites once with plain greedy decoding (eval.py, arm practice_base or
practice_vad under runs/whisper/tune/), keeping each clip's openai statistics: no-speech probability, average
log-prob, compression ratio. Step 2 is an offline grid over the threshold rule in common.threshold_keep:
  no_speech        skip if nsp > no_speech and avg_logprob <= logprob   (openai-whisper's rule)
  logprob          the log-prob condition of that rule (None = skip on nsp alone)
  drop_logprob     strict extra: drop if avg_logprob < drop_logprob
  drop_compression strict extra: drop if compression ratio > drop_compression
It picks the config with the lowest practice non-speech hallucination rate that keeps, relative to plain decoding
("thresholds at matched WER", mirroring the held-out pass bar): practice clean WER within +0.3 points (--wer-tol),
practice speech-in-background WER within +1.0 point (--bg-tol) with at most +1 point of fully dropped utterances,
and practice padded insertions within +0.2 points. Ties go to lower padded insertions, then to fewer knobs.
Two files are written: thresholds.json (openai's rule only: no_speech + logprob; the faithful arm 2) and
thresholds_strict.json (the same plus the strict drops; a stronger, less standard variant). With --vad the names
are thresholds_vad.json / thresholds_vad_strict.json.

Notes: HF transformers 5.x does apply these thresholds to short-form input, but only through temperature fallback
(stochastic re-sampling) plus the no-speech skip; we implement the rule deterministically on one greedy pass so the
grid is exact and offline. condition_on_previous_text is moot: every clip is a single <=30 s window, so there is no
previous text to condition on.
"""
import argparse
import itertools
import json
import subprocess
import sys

import common as C

GRID = dict(no_speech=[None, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
            logprob=[None, -0.2, -0.4, -0.6, -0.8, -1.0, -1.5],
            drop_logprob=[None, -0.6, -0.8, -1.0, -1.25, -1.5],
            drop_compression=[None, 2.4, 2.0, 1.8])


def load_rows(arm_dir, suite):
    return [json.loads(l) for l in (arm_dir / f"{suite}.jsonl").read_text(encoding="utf-8").splitlines()]


def _wstats(refs_n, hyps_n):
    import jiwer
    ins = dele = sub = nref = 0
    anyins, dropped = [], []
    for r, h in zip(refs_n, hyps_n):
        o = jiwer.process_words(r, h)
        ins, dele, sub, nref = ins + o.insertions, dele + o.deletions, sub + o.substitutions, nref + len(r.split())
        anyins.append(o.insertions > 0)
        dropped.append(not h)
    return (100 * (ins + dele + sub) / nref, 100 * ins / nref, sum(anyins) / len(anyins),
            sum(dropped) / len(dropped))


_cache = {}


def score(rows, cfg):
    """Practice metrics for one threshold config. Normalized texts and per-clip alignments are cached, since a
    config only decides, per clip, between the raw transcript and an empty one."""
    if not _cache:
        for s in C.PRACTICE_SUITES:
            _cache[s] = [dict(r, n=C.norm(r["raw_text"]) if "nsp" in r else "", rn=C.norm(r.get("ref") or ""))
                         for r in rows[s]]
    keep = {s: [("nsp" in r) and C.threshold_keep(r, cfg) for r in _cache[s]] for s in C.PRACTICE_SUITES}
    ns = _cache["practice_nonspeech"]
    nonempty = [k and bool(r["n"]) for r, k in zip(ns, keep["practice_nonspeech"])]
    by_cat = {c: sum(e for e, r in zip(nonempty, ns) if r["category"] == c) /
              max(sum(r["category"] == c for r in ns), 1) for c in C.CATEGORIES}
    out = dict(halluc=round(sum(nonempty) / len(ns), 4), halluc_by_cat={k: round(v, 4) for k, v in by_cat.items()})
    out["halluc_notag"] = round(sum(e and not C.is_sound_tag(r["n"]) for e, r in zip(nonempty, ns)) / len(ns), 4)
    for s, pre in (("practice_speech", ""), ("practice_padded", "padded_"), ("practice_bg", "bg_")):
        rr = [r for r in _cache[s] if r["rn"]]
        kk = [k for r, k in zip(_cache[s], keep[s]) if r["rn"]]
        key = (s, tuple(kk))
        if key not in _cache:
            _cache[key] = _wstats([r["rn"] for r in rr], [r["n"] if k else "" for r, k in zip(rr, kk)])
        wer, ins, anyins, dropped = _cache[key]
        out.update({f"{pre}wer": round(wer, 3), f"{pre}ins": round(ins, 3), f"{pre}any_ins": round(anyins, 4),
                    f"{pre}dropped": round(dropped, 4)})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vad", action="store_true")
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--edit", default=None, help="tune on top of an edit (not used by the planned arms)")
    ap.add_argument("--wer-tol", type=float, default=0.3)
    ap.add_argument("--bg-tol", type=float, default=1.0)
    ap.add_argument("--rerun", action="store_true", help="re-transcribe even if practice transcripts exist")
    a = ap.parse_args()
    arm = "practice_vad" if a.vad else "practice_base"
    if a.edit:
        arm += "_edit"
    tune_dir = C.RUNS / "tune"
    cmd = [sys.executable, str(C.ROOT / "whisper_halluc" / "eval.py"), "--arm", arm, "--model", a.model,
           "--suites", ",".join(C.PRACTICE_SUITES), "--out", str(tune_dir)]
    if a.vad:
        cmd.append("--vad")
    if a.edit:
        cmd += ["--edit", a.edit]
    if not all((tune_dir / arm / f"{s}.jsonl").exists() for s in C.PRACTICE_SUITES) or a.rerun:
        subprocess.run(cmd, check=True)
    rows = {s: load_rows(tune_dir / arm, s) for s in C.PRACTICE_SUITES}
    plain = score(rows, {})
    openai = score(rows, C.OPENAI_DEFAULT_THRESHOLDS)
    results = []
    keys = list(GRID)
    for vals in itertools.product(*GRID.values()):
        cfg = dict(zip(keys, vals))
        if cfg["no_speech"] is None and cfg["logprob"] is not None:
            continue  # logprob only matters as part of the no-speech rule
        s = score(rows, cfg)
        s["config"] = cfg
        s["knobs"] = sum(v is not None for v in vals)
        s["strict"] = cfg["drop_logprob"] is not None or cfg["drop_compression"] is not None
        results.append(s)
    # matched WER, mirroring the held-out pass bar: clean WER <= +wer_tol, speech-in-background WER <= +bg_tol and
    # no more than +1 point of fully dropped utterances, padded insertions <= +0.2 points
    ok = [r for r in results if r["wer"] <= plain["wer"] + a.wer_tol
          and r["bg_wer"] <= plain["bg_wer"] + a.bg_tol and r["bg_dropped"] <= plain["bg_dropped"] + 0.01 + 1e-9
          and r["padded_ins"] <= plain["padded_ins"] + 0.2]
    ok.sort(key=lambda r: (r["halluc"], r["padded_ins"], r["knobs"], r["wer"]))
    base_name = "thresholds_vad" if a.vad else "thresholds"
    if a.edit:
        base_name += "_edit"
    for family, cand in (("", [r for r in ok if not r["strict"]]), ("_strict", ok)):
        best = cand[0]
        out = dict(config=best["config"], family="openai rule only (no_speech + logprob)" if not family else
                   "openai rule plus strict drops (drop_logprob, drop_compression)",
                   practice=best, practice_plain=plain, practice_openai_default=openai, wer_tol=a.wer_tol,
                   vad=a.vad, model=a.model, n_grid=len(results), top10=cand[:10])
        name = f"{base_name}{family}.json"
        (C.RUNS / name).write_text(json.dumps(out, indent=1))
        print(name, json.dumps(dict(best=best), indent=1))
        print(f"wrote {C.RUNS / name}")
    print(json.dumps(dict(plain=plain, openai_default=openai), indent=1))


if __name__ == "__main__":
    main()
