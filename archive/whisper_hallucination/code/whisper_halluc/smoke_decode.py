"""Smoke test: our greedy decoder == HF model.generate (default short-form decoding) on a few clips, and whether HF
5.x applies no_speech_threshold / logprob_threshold / compression_ratio_threshold to short-form input.
  python whisper_halluc/smoke_decode.py [--model openai/whisper-small] [--n 12]"""
import argparse
import json
import time

import torch

import common as C
from gpu_lock import gpu


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    proc = C.load_processor(a.model)
    model = C.load_model(a.model)
    tok = C.Tok(model, proc)
    clips = []
    for suite in ("practice_nonspeech", "practice_speech", "practice_padded"):
        man, aud = C.load_suite(suite)
        step = max(len(man) // a.n, 1)
        clips += [(man[i]["id"], aud[i]) for i in range(0, len(man), step)][:a.n]
    audios = [x for _, x in clips]
    report = {}
    with gpu("whisper smoke_decode"):
        C.gpu_setup()
        model.cuda()
        t0 = time.time()
        ours = C.transcribe(model, proc, tok, audios, batch=16)
        t_ours = time.time() - t0
        f = C.features(proc, audios, "cuda")
        t0 = time.time()
        hf = []
        for i in range(0, len(audios), 16):
            g = model.generate(input_features=f[i:i + 16], language="en", task="transcribe")
            hf += proc.batch_decode(g, skip_special_tokens=True)
        t_hf = time.time() - t0
        # HF thresholds on short-form: does a no-speech skip / fallback happen?
        th = []
        for i in range(0, len(audios), 16):
            g = model.generate(input_features=f[i:i + 16], language="en", task="transcribe",
                               no_speech_threshold=0.6, logprob_threshold=-1.0, compression_ratio_threshold=1.35,
                               temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
            g = g.sequences if hasattr(g, "sequences") else g
            g = g.long()
            g[g < 0] = tok.eot
            th += proc.batch_decode(g, skip_special_tokens=True)
        model.cpu()
    mism = [(cid, o["text"], h) for (cid, _), o, h in zip(clips, ours, hf) if o["text"].strip() != h.strip()]
    changed = [(cid, h, t, round(o["nsp"], 3), round(o["avg_logprob"], 3), round(o["compression_ratio"], 2))
               for (cid, _), o, h, t in zip(clips, ours, hf, th) if h.strip() != t.strip()]
    report = dict(n=len(clips), mismatches=len(mism), mismatch_examples=mism[:5], t_ours=round(t_ours, 1),
                  t_hf=round(t_hf, 1), hf_threshold_changes=len(changed), hf_threshold_examples=changed[:8],
                  examples=[(cid, o["text"][:80], round(o["nsp"], 3), round(o["avg_logprob"], 2))
                            for (cid, _), o in zip(clips, ours)][:40])
    print(json.dumps(report, indent=1, ensure_ascii=False))
    (C.RUNS / "smoke").mkdir(parents=True, exist_ok=True)
    (C.RUNS / "smoke" / "decode_check.json").write_text(json.dumps(report, indent=1, ensure_ascii=False),
                                                       encoding="utf-8")


if __name__ == "__main__":
    main()
