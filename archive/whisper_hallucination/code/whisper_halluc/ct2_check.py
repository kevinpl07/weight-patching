"""Sanity check: an edited checkpoint survives conversion to CTranslate2 (the faster-whisper runtime).

  python whisper_halluc/ct2_check.py --edit runs/whisper/session_1/submitted_edit.safetensors --name session_1
  python whisper_halluc/ct2_check.py --edit runs/whisper/head_scan/edit.safetensors --name head_scan

Saves the edited HF model, converts it with ctranslate2's TransformersConverter (float32, no quantization), then
transcribes --n practice clips with both runtimes on the CPU (no GPU lock needed) using the same greedy decoding
(prompt sot/en/transcribe/notimestamps, beam 1, suppress_blank, the model's suppress list) and reports how many
transcripts match exactly, plus the unedited HF transcript for contrast. Output: runs/whisper/ct2/<name>/check.json.
"""
import argparse
import json
import shutil

import numpy as np
import torch

import common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--edit", default=None)
    ap.add_argument("--name", default="check")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--keep", action="store_true", help="keep the converted model folders")
    a = ap.parse_args()
    import ctranslate2
    from ctranslate2.converters import TransformersConverter

    out = C.RUNS / "ct2" / a.name
    hf_dir, ct_dir = out / "hf", out / "ct2"
    out.mkdir(parents=True, exist_ok=True)
    proc = C.load_processor(a.model)
    model = C.load_model(a.model, edit=C.read_edit(a.edit) if a.edit else None)
    model.save_pretrained(hf_dir)
    proc.save_pretrained(hf_dir)
    TransformersConverter(str(hf_dir)).convert(str(ct_dir), force=True)
    ct = ctranslate2.models.Whisper(str(ct_dir), device="cpu", compute_type="float32")

    clips = []
    for suite, k in (("practice_nonspeech", a.n * 3 // 4), ("practice_speech", a.n - a.n * 3 // 4)):
        man, aud = C.load_suite(suite)
        step = max(len(man) // max(k, 1), 1)
        clips += [(man[i]["id"], np.array(aud[i])) for i in range(0, len(man), step)][:k]
    tok = C.Tok(model, proc)
    hf = C.transcribe(model, proc, tok, [x for _, x in clips], batch=4, device="cpu")
    base = C.load_model(a.model)
    hb = C.transcribe(base, proc, tok, [x for _, x in clips], batch=4, device="cpu") if a.edit else hf
    rows = []
    for (cid, x), h, b in zip(clips, hf, hb):
        f = C.features(proc, [x]).numpy()
        r = ct.generate(ctranslate2.StorageView.from_array(np.ascontiguousarray(f)), [tok.prompt], beam_size=1,
                        max_length=C.MAX_POSITIONS, suppress_blank=True, suppress_tokens=[-1],
                        return_no_speech_prob=True)[0]
        ids = r.sequences_ids[0]
        text = tok.text([i for i in ids if i not in tok.prompt])
        rows.append(dict(id=cid, hf_edited=h["text"], ct2_edited=text, hf_base=b["text"],
                         match=text.strip() == h["text"].strip(), ct2_nsp=round(r.no_speech_prob, 4),
                         hf_nsp=round(h["nsp"], 4)))
    res = dict(edit=a.edit, model=a.model, n=len(rows), matches=sum(r["match"] for r in rows),
               ctranslate2=ctranslate2.__version__, rows=rows)
    (out / "check.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"{res['matches']}/{res['n']} transcripts identical between HF and CTranslate2 ({out / 'check.json'})")
    for r in rows:
        if not r["match"]:
            print("  differs:", r["id"], repr(r["hf_edited"][:60]), "vs", repr(r["ct2_edited"][:60]))
    if not a.keep:
        shutil.rmtree(hf_dir, ignore_errors=True)
        shutil.rmtree(ct_dir, ignore_errors=True)


if __name__ == "__main__":
    torch.set_num_threads(8)
    main()
