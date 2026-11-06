"""Score one arm on the held-out suites (or any suites). An arm = model + optional edit + decoding config + optional VAD.

  python whisper_halluc/eval.py --arm base
  python whisper_halluc/eval.py --arm thresholds     --thresholds runs/whisper/thresholds.json
  python whisper_halluc/eval.py --arm vad            --vad
  python whisper_halluc/eval.py --arm vad_thresholds --vad --thresholds runs/whisper/thresholds_vad.json
  python whisper_halluc/eval.py --arm head_scan      --edit runs/whisper/head_scan/edit.safetensors
  python whisper_halluc/eval.py --arm agent1         --edit runs/whisper/session_1/submitted_edit.safetensors
  python whisper_halluc/eval.py --arm agent1_vad     --edit ... --vad
  python whisper_halluc/eval.py --arm thresholds --thresholds runs/whisper/thresholds.json --from-arm base  (no GPU)
  options: --suites nonspeech,padded,wer  --limit N  --model openai/whisper-large-v3-turbo  --batch 16
           --chunk 150 (clips per GPU-lock hold)  --out runs/whisper/eval  --fresh (ignore saved transcripts)

Every transcript is appended to <out>/<arm>/<suite>.jsonl as soon as its chunk finishes, so a killed run resumes
where it stopped. <out>/<arm>/summary.json holds the config, metrics per suite and GPU seconds.
Decoding: greedy, language=en, task=transcribe, no timestamps, HF default suppress tokens (common.greedy_decode,
checked against model.generate by smoke_decode.py). Thresholds use the openai rule in common.threshold_keep.
VAD: Silero, faster-whisper defaults; no speech -> empty transcript, else the speech regions are concatenated and
transcribed (faster-whisper's vad_filter behaviour). VAD segments are cached in runs/whisper/vad_cache/.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

import common as C
from gpu_lock import gpu

SUITE_SIZES = dict(nonspeech=300, padded=300, wer=500, bg=300, other=300, practice_nonspeech=100,
                   practice_speech=100, practice_padded=50, practice_bg=50)


def vad_cache(suite, man, audio, opts=None):
    key = hashlib.md5(json.dumps(dict(C.VAD_OPTS, **(opts or {})), sort_keys=True).encode()).hexdigest()[:8]
    path = C.RUNS / "vad_cache" / f"{suite}_{key}.json"
    cache = json.loads(path.read_text()) if path.exists() else {}
    todo = [i for i, m in enumerate(man) if m["id"] not in cache]
    if todo:
        t0 = time.time()
        for i in todo:
            cache[man[i]["id"]] = C.vad_segments(audio[i], opts)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(cache))
        print(f"  VAD on {len(todo)} clips of {suite}: {time.time() - t0:.0f} s CPU")
    return cache


def load_thresholds(spec):
    if not spec:
        return None
    if spec == "openai":
        return dict(C.OPENAI_DEFAULT_THRESHOLDS)
    d = json.loads(Path(spec).read_text())
    return d.get("config", d)


def from_arm(a, suite, man, thr, path):
    """Re-score another arm's saved transcripts under a threshold config (no GPU). The source arm must have the same
    model, edit and VAD setting; its raw transcripts and statistics are reused."""
    src = Path(a.out) / a.from_arm
    cfg = json.loads((src / "summary.json").read_text())["config"]
    if cfg["vad"] != a.vad or cfg["edit"] != a.edit or cfg["model"] != a.model:
        raise SystemExit(f"--from-arm {a.from_arm} has vad={cfg['vad']} edit={cfg['edit']} model={cfg['model']}")
    rows = {}
    for line in (src / f"{suite}.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        rows[r["id"]] = r
    missing = [m["id"] for m in man if m["id"] not in rows]
    if missing:
        raise SystemExit(f"--from-arm {a.from_arm} lacks {len(missing)} clips of {suite}; finish that arm first")
    out = []
    with open(path, "w", encoding="utf-8") as f:
        for m in man:
            r = dict(rows[m["id"]])
            if "nsp" in r:
                keep = C.threshold_keep(r, thr) if thr else True
                r.update(text=r["raw_text"] if keep else "", threshold_dropped=not keep)
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            out.append(r)
    return C.suite_metrics(suite, out), 0.0, 0


def run_suite(a, suite, model, proc, tok, thr, out_dir):
    man, audio = C.load_suite(suite)
    if a.limit:
        man, audio = man[:a.limit], audio[:a.limit]
    path = out_dir / f"{suite}.jsonl"
    done = {}
    if path.exists() and not a.fresh:
        for line in path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            done[r["id"]] = r
    elif path.exists():
        path.unlink()
    if a.from_arm:
        return from_arm(a, suite, man, thr, path)
    segs = vad_cache(suite, man, audio) if a.vad else {}
    todo = [i for i, m in enumerate(man) if m["id"] not in done]
    gpu_s = 0.0
    for c0 in range(0, len(todo), a.chunk):
        idx = todo[c0:c0 + a.chunk]
        inputs, rows = [], []
        for i in idx:
            m = man[i]
            row = {k: m[k] for k in ("id", "category", "sub", "ref", "pad", "bg", "snr", "seconds") if k in m}
            x = audio[i]
            if a.vad:
                sg = segs[m["id"]]
                row["vad_speech_s"] = round(sum(e - s for s, e in sg) / C.SR, 2)
                row["vad_segments"] = len(sg)
                x = C.vad_collect(x, sg)
                row["vad_skipped"] = x is None
            rows.append(row)
            inputs.append(x)
        run_idx = [j for j, x in enumerate(inputs) if x is not None]
        res = {}
        if run_idx:
            with gpu(f"whisper eval {a.arm} {suite} {c0}/{len(todo)}"):
                C.gpu_setup()
                t0 = time.time()
                model.cuda()
                out = C.transcribe(model, proc, tok, [inputs[j] for j in run_idx], batch=a.batch,
                                   max_new=a.max_new)
                model.cpu()
                torch.cuda.empty_cache()
                gpu_s += time.time() - t0
            res = dict(zip(run_idx, out))
        with open(path, "a", encoding="utf-8") as f:
            for j, row in enumerate(rows):
                r = res.get(j)
                if r is None:
                    row.update(raw_text="", text="", n_tokens=0)
                else:
                    keep = C.threshold_keep(r, thr) if thr else True
                    row.update(raw_text=r["text"], text=r["text"] if keep else "", n_tokens=len(r["tokens"]),
                               nsp=round(r["nsp"], 5), avg_logprob=round(r["avg_logprob"], 4),
                               compression_ratio=round(r["compression_ratio"], 3),
                               p_eot_first=round(r["p_eot_first"], 5), hit_max=r["hit_max"],
                               threshold_dropped=not keep)
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
                done[row["id"]] = row
        print(f"  {suite}: {len(done)}/{len(man)} clips, {gpu_s:.0f} s GPU so far", flush=True)
    rows = [done[m["id"]] for m in man]
    met = C.suite_metrics(suite, rows)
    return met, gpu_s, len(todo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True)
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--edit", default=None)
    ap.add_argument("--thresholds", default=None, help="JSON file from tune_thresholds.py, or 'openai'")
    ap.add_argument("--vad", action="store_true")
    ap.add_argument("--suites", default="nonspeech,padded,wer,bg,other")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--chunk", type=int, default=150)
    ap.add_argument("--max-new", dest="max_new", type=int, default=None)
    ap.add_argument("--dtype", default="fp32", choices=["fp32", "fp16"])
    ap.add_argument("--out", default=str(C.RUNS / "eval"))
    ap.add_argument("--fresh", action="store_true")
    ap.add_argument("--from-arm", dest="from_arm", default=None,
                    help="re-score this arm's saved transcripts with --thresholds instead of transcribing (no GPU)")
    a = ap.parse_args()
    out_dir = Path(a.out) / a.arm
    out_dir.mkdir(parents=True, exist_ok=True)
    thr = load_thresholds(a.thresholds)
    dtype = torch.float16 if a.dtype == "fp16" else torch.float32
    proc = C.load_processor(a.model)
    model = C.load_model(a.model, dtype=dtype)  # (unused with --from-arm, but cheap)
    base_params = {k: v.detach().clone() for k, v in C.param_dict(model).items()} if a.edit else None
    n_changed = C.apply_edit(model, a.edit) if a.edit else 0
    tok = C.Tok(model, proc)
    cfg = dict(arm=a.arm, from_arm=a.from_arm, model=a.model, edit=a.edit, edit_entries_changed=n_changed, thresholds=thr, vad=a.vad,
               vad_opts=C.VAD_OPTS if a.vad else None, dtype=a.dtype, limit=a.limit,
               decoding="greedy, en, transcribe, notimestamps, HF default suppress/begin_suppress, max 448 positions")
    if a.edit:
        cfg["edit_stats"] = {k: dict(changed=n, l2=round(l2, 4))
                             for k, (n, l2) in C.edit_stats(C.read_edit(a.edit), base_params).items()}
        del base_params
    summ_path = out_dir / "summary.json"
    summ = json.loads(summ_path.read_text()) if summ_path.exists() and not a.fresh else {}
    summ["config"] = cfg
    summ.setdefault("suites", {})
    summ.setdefault("gpu_seconds", {})
    for suite in a.suites.split(","):
        print(f"[{a.arm}] {suite}", flush=True)
        met, gs, n_new = run_suite(a, suite, model, proc, tok, thr, out_dir)
        summ["suites"][suite] = met
        if n_new:
            summ["gpu_seconds"][suite] = dict(seconds=round(gs, 1), clips=n_new,
                                              per_clip=round(gs / n_new, 3),
                                              projected_full_min=round(gs / n_new * SUITE_SIZES.get(suite, n_new) / 60,
                                                                       2))
        summ_path.write_text(json.dumps(summ, indent=1))
        print(json.dumps({suite: met}, indent=1))
    proj = sum(v["projected_full_min"] for k, v in summ["gpu_seconds"].items() if k in C.HELDOUT_SUITES)
    summ["projected_gpu_minutes_heldout"] = round(proj, 1)
    summ_path.write_text(json.dumps(summ, indent=1))
    print(f"wrote {summ_path}; projected GPU minutes for the held-out suites: {proj:.1f}")


if __name__ == "__main__":
    main()
