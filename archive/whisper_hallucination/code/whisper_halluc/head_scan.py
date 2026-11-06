"""Head-ablation scan (arm 5, the strongest script arm), on PRACTICE data only.

  python whisper_halluc/head_scan.py                      # self- and cross-attention heads of every decoder layer
  python whisper_halluc/head_scan.py --kinds self         # decoder self-attention heads only (half the time)
  -> runs/whisper/head_scan/single.jsonl   one line per ablated head (resumable)
     runs/whisper/head_scan/topk.json      the top-k curve and the chosen k
     runs/whisper/head_scan/edit.safetensors   the chosen top-k ablation as an ordinary edit file

Ablating head h of a decoder attention block = zeroing columns [h*d_head, (h+1)*d_head) of that block's
out_proj.weight, so the head's output never reaches the residual stream. That is a pure weight edit (the out_proj
bias is untouched), so the result drops into any runtime.

1. Score the base model and then each single-head ablation on practice_nonspeech (hallucination rate = share of
   non-empty normalized transcripts, decoding capped at --ns-max-new tokens because only emptiness matters; plus the
   mean normalized word count as a tie-breaker), practice_speech (clean WER) and practice_bg (speech in background:
   WER and share of fully dropped utterances), with full decoding of speech.
2. Rank heads by the drop in hallucination rate among heads whose single ablation raises clean practice WER by at
   most --screen-wer (default 1.0 point) and speech-in-background WER by at most twice that; ties broken by the drop
   in word count.
3. Ablate the top k heads together for k = 1..--kmax and score each on all four practice suites. Choose the k with
   the lowest hallucination rate that keeps (mirroring the held-out pass bar) clean WER within +0.3 (--wer-tol),
   background WER within +1.0 (--bg-tol) with at most +1 point more fully dropped utterances, and padded insertions
   within +0.2 points; ties -> smaller k.

Encoder outputs are computed once per GPU-lock hold (decoder-only ablations do not change them). Each lock hold
covers --per-lock ablations (~2-4 min); between holds other jobs can use the GPU.
"""
import argparse
import json
import time

import numpy as np
import torch

import common as C
from gpu_lock import gpu

OUT = C.RUNS / "head_scan"


class Scorer:
    def __init__(self, model, proc, tok, a):
        self.model, self.proc, self.tok, self.a = model, proc, tok, a
        man, aud = C.load_suite("practice_nonspeech")
        self.ns = (man, list(aud))
        man, aud = C.load_suite("practice_speech")
        o = np.argsort([m["n"] for m in man])          # sort by length so batches finish together
        self.sp = ([man[i] for i in o], [aud[i] for i in o])
        man, aud = C.load_suite("practice_padded")
        o = np.argsort([m["n"] for m in man])
        self.pp = ([man[i] for i in o], [aud[i] for i in o])
        man, aud = C.load_suite("practice_bg")
        o = np.argsort([m["n"] for m in man])
        self.bg = ([man[i] for i in o], [aud[i] for i in o])
        self.feats = {k: C.features(proc, v[1]) for k, v in (("ns", self.ns), ("sp", self.sp), ("pp", self.pp),
                                                              ("bg", self.bg))}
        self.enc = {}

    def encode(self):
        """Encoder outputs on the GPU for all practice clips (call inside the lock, after model.cuda())."""
        self.enc = {}
        for k, f in self.feats.items():
            outs = [C.encode(self.model, f[i:i + 32].cuda()) for i in range(0, len(f), 32)]
            self.enc[k] = torch.cat(outs)

    def decode(self, key, max_new=None):
        enc = self.enc[key]
        res = []
        for i in range(0, len(enc), self.a.batch):
            res += C.greedy_decode(self.model, self.tok, enc[i:i + self.a.batch], max_new=max_new)
        return [self.tok.text(r["tokens"]) for r in res], res

    def score(self, padded=False):
        ns_text, ns_res = self.decode("ns", self.a.ns_max_new)
        norm = [C.norm(t) for t in ns_text]
        cats = [m["category"] for m in self.ns[0]]
        out = dict(halluc=float(np.mean([bool(n) for n in norm])),
                   halluc_notag=float(np.mean([bool(n) and not C.is_sound_tag(n) for n in norm])),
                   ns_words=float(np.mean([len(n.split()) for n in norm])),
                   ns_nsp=float(np.mean([r["nsp"] for r in ns_res])),
                   by_cat={c: float(np.mean([bool(n) for n, cc in zip(norm, cats) if cc == c])) for c in C.CATEGORIES})
        sp_text, _ = self.decode("sp", self.a.sp_max_new)
        out["wer"] = C.word_stats([m["ref"] for m in self.sp[0]], sp_text)["wer"]
        bg_text, _ = self.decode("bg", self.a.sp_max_new)
        w = C.word_stats([m["ref"] for m in self.bg[0]], bg_text)
        out.update(bg_wer=w["wer"], bg_dropped=w["dropped"])
        if padded:
            pp_text, _ = self.decode("pp", None)
            w = C.word_stats([m["ref"] for m in self.pp[0]], pp_text)
            out.update(padded_ins=w["ins_rate"], padded_any_ins=w["any_ins"], padded_wer=w["wer"])
        return out


def head_slices(model, kinds):
    cfg = model.config
    H = cfg.decoder_attention_heads
    hd = cfg.d_model // H
    out = []
    for kind in kinds:
        attr = "self_attn" if kind == "self" else "encoder_attn"
        for l in range(cfg.decoder_layers):
            name = f"model.decoder.layers.{l}.{attr}.out_proj.weight"
            for h in range(H):
                out.append(dict(kind=kind, layer=l, head=h, param=name, cols=[h * hd, (h + 1) * hd]))
    return out


class Ablate:
    """Context manager: zero the given heads' out_proj columns, restore on exit."""

    def __init__(self, model, heads):
        self.p = C.param_dict(model)
        self.heads = heads
        self.saved = []

    def __enter__(self):
        with torch.no_grad():
            for h in self.heads:
                w = self.p[h["param"]]
                c0, c1 = h["cols"]
                self.saved.append((w, c0, c1, w[:, c0:c1].clone()))
                w[:, c0:c1] = 0
        return self

    def __exit__(self, *exc):
        with torch.no_grad():
            for w, c0, c1, v in reversed(self.saved):
                w[:, c0:c1] = v
        self.saved = []


def hid(h):
    return f"{h['kind']}:L{h['layer']}:H{h['head']}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--kinds", default="self,cross")
    ap.add_argument("--kmax", type=int, default=10)
    ap.add_argument("--wer-tol", type=float, default=0.3)
    ap.add_argument("--bg-tol", type=float, default=1.0)
    ap.add_argument("--screen-wer", type=float, default=1.0)
    ap.add_argument("--ns-max-new", dest="ns_max_new", type=int, default=24)
    ap.add_argument("--sp-max-new", dest="sp_max_new", type=int, default=200)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--per-lock", type=int, default=16)
    ap.add_argument("--pause", type=float, default=20.0, help="seconds to wait between lock holds")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--max-heads", dest="max_heads", type=int, default=None, help="smoke test: scan only the first N")
    a = ap.parse_args()
    out = C.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    proc = C.load_processor(a.model)
    model = C.load_model(a.model)
    tok = C.Tok(model, proc)
    S = Scorer(model, proc, tok, a)
    heads = head_slices(model, a.kinds.split(","))
    if a.max_heads:
        heads = heads[:a.max_heads]
    single_path = out / "single.jsonl"
    done = {}
    if single_path.exists():
        for line in single_path.read_text().splitlines():
            r = json.loads(line)
            done[r["id"]] = r
    todo = ([None] if "base" not in done else []) + [h for h in heads if hid(h) not in done]
    t_start = time.time()
    gpu_s = 0.0
    for c0 in range(0, len(todo), a.per_lock):
        chunk = todo[c0:c0 + a.per_lock]
        with gpu(f"whisper head_scan {c0}/{len(todo)}"):
            C.gpu_setup()
            t0 = time.time()
            model.cuda()
            S.encode()
            lines = []
            for h in chunk:
                if h is None:
                    r = dict(id="base", **S.score(padded=True))
                else:
                    with Ablate(model, [h]):
                        r = dict(id=hid(h), **{k: h[k] for k in ("kind", "layer", "head")}, **S.score())
                lines.append(r)
            model.cpu()
            S.enc = {}
            torch.cuda.empty_cache()
            gpu_s += time.time() - t0
        time.sleep(a.pause)  # let other jobs take the lock between holds
        with open(single_path, "a") as f:
            for r in lines:
                f.write(json.dumps(r) + "\n")
                done[r["id"]] = r
        print(f"  scanned {min(c0 + a.per_lock, len(todo))}/{len(todo)}; {gpu_s / 60:.1f} GPU min; "
              f"last {lines[-1]['id']} halluc {lines[-1]['halluc']:.2f} wer {lines[-1]['wer']:.2f}", flush=True)

    base = done["base"]
    singles = [done[hid(h)] for h in heads]
    for r in singles:
        r["d_halluc"] = r["halluc"] - base["halluc"]
        r["d_words"] = r["ns_words"] - base["ns_words"]
        r["d_wer"] = r["wer"] - base["wer"]
        r["d_bg_wer"] = r["bg_wer"] - base["bg_wer"]
    ok = [r for r in singles if r["d_wer"] <= a.screen_wer and r["d_bg_wer"] <= 2 * a.screen_wer]
    ok.sort(key=lambda r: (r["d_halluc"], r["d_words"], r["d_wer"]))
    ranked = ok[:a.kmax]
    print("top single-head ablations:")
    for r in ranked:
        print(f"  {r['id']:16s} d_halluc {r['d_halluc']:+.3f} d_words {r['d_words']:+.2f} d_wer {r['d_wer']:+.2f}")
    by_id = {hid(h): h for h in heads}

    curve_path = out / "topk.json"
    curve = json.loads(curve_path.read_text()).get("curve", []) if curve_path.exists() else []
    curve = [c for c in curve if c["heads"] == [r["id"] for r in ranked[:c["k"]]]]
    have = {c["k"] for c in curve}
    ks = [k for k in range(1, len(ranked) + 1) if k not in have]
    for c0 in range(0, len(ks), max(a.per_lock // 4, 1)):
        with gpu(f"whisper head_scan topk {ks[c0]}"):
            C.gpu_setup()
            t0 = time.time()
            model.cuda()
            S.encode()
            for k in ks[c0:c0 + max(a.per_lock // 4, 1)]:
                hs = [by_id[r["id"]] for r in ranked[:k]]
                with Ablate(model, hs):
                    s = S.score(padded=True)
                curve.append(dict(k=k, heads=[hid(h) for h in hs], **s))
                print(f"  top-{k}: halluc {s['halluc']:.3f} wer {s['wer']:.2f} bg_wer {s['bg_wer']:.2f} "
                      f"bg_dropped {s['bg_dropped']:.2f} padded_ins {s['padded_ins']:.2f}",
                      flush=True)
            model.cpu()
            S.enc = {}
            gpu_s += time.time() - t0
        curve_path.write_text(json.dumps(dict(base=base, curve=sorted(curve, key=lambda c: c["k"])), indent=1))
    curve.sort(key=lambda c: c["k"])
    good = [c for c in curve if c["wer"] <= base["wer"] + a.wer_tol and c["bg_wer"] <= base["bg_wer"] + a.bg_tol
            and c["bg_dropped"] <= base["bg_dropped"] + 0.01 + 1e-9 and c["padded_ins"] <= base["padded_ins"] + 0.2]
    best = min(good, key=lambda c: (c["halluc"], c["k"])) if good else None
    res = dict(base=base, curve=curve, chosen=best, wer_tol=a.wer_tol, ranked=ranked, model=a.model,
               kinds=a.kinds, gpu_minutes_this_run=round(gpu_s / 60, 1),
               wall_minutes_this_run=round((time.time() - t_start) / 60, 1))
    curve_path.write_text(json.dumps(res, indent=1))
    if best is None:
        print("no k keeps practice WER within tolerance; no edit written")
        return
    hs = [by_id[i] for i in best["heads"]]
    with Ablate(model, hs):
        p = C.param_dict(model)
        edit = {n: p[n].detach().clone() for n in sorted({h["param"] for h in hs})}
    C.write_edit(out / "edit.safetensors", edit)
    print(f"chosen k={best['k']}: {best['heads']}; practice halluc {base['halluc']:.3f} -> {best['halluc']:.3f}, "
          f"WER {base['wer']:.2f} -> {best['wer']:.2f}; wrote {out / 'edit.safetensors'}")


if __name__ == "__main__":
    main()
