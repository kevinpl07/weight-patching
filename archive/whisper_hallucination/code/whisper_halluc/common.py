"""Shared helpers for the Whisper non-speech hallucination experiment (experiment 3 of
PLAN_2026-11-04_practical_edits.md).

- paths and suites: 16 kHz mono float32 clips in data_audio/derived/<suite>.npy plus <suite>.json manifests
- model loading, edit files ({param_name: full new tensor} in safetensors) and the shared apply-edit helper
- a greedy decoder that reproduces HF transformers' default short-form Whisper decoding (prompt
  <|startoftranscript|><|en|><|transcribe|><|notimestamps|>, suppress_tokens, begin_suppress_tokens, greedy,
  max 448 positions) and also returns the openai-whisper statistics: no-speech probability, average log-prob,
  gzip compression ratio
- the openai-whisper threshold rule, Silero VAD front-end, and the metrics (hallucination, insertion, WER)
"""
import json
import math
import sys
import zlib
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DATA = ROOT / "data_audio"
RAW = DATA / "raw"
DERIVED = DATA / "derived"
RUNS = ROOT / "runs" / "whisper"
SR = 16000
DEFAULT_MODEL = "openai/whisper-small"
MEM_FRAC = 0.85
MAX_POSITIONS = 448            # decoder max_target_positions; HF stops at max_length=448 total tokens
CATEGORIES = ("silence", "noise", "environmental", "music")
HELDOUT_SUITES = ("nonspeech", "padded", "wer", "bg", "other")
PRACTICE_SUITES = ("practice_nonspeech", "practice_speech", "practice_padded", "practice_bg")


# ----------------------------------------------------------------------------------------------- data
def load_suite(name):
    """Returns (manifest list of dicts, list of float32 arrays). The arrays are views into a memory map."""
    man = json.loads((DERIVED / f"{name}.json").read_text(encoding="utf-8"))
    buf = np.load(DERIVED / f"{name}.npy", mmap_mode="r")
    audio = [buf[m["offset"]:m["offset"] + m["n"]] for m in man]
    return man, audio


def suite_exists(name):
    return (DERIVED / f"{name}.json").exists() and (DERIVED / f"{name}.npy").exists()


# ----------------------------------------------------------------------------------------------- model
def gpu_setup():
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(MEM_FRAC)
        torch.backends.cuda.matmul.allow_tf32 = False  # keep fp32 exact so tiny edits are not rounded away


def load_processor(model_name=DEFAULT_MODEL):
    from transformers import WhisperProcessor
    return WhisperProcessor.from_pretrained(model_name)


def load_model(model_name=DEFAULT_MODEL, edit=None, dtype=torch.float32, attn="sdpa"):
    """WhisperForConditionalGeneration on CPU in eval mode, with an optional edit applied."""
    from transformers import WhisperForConditionalGeneration
    m = WhisperForConditionalGeneration.from_pretrained(model_name, dtype=dtype, attn_implementation=attn)
    m.eval()
    if edit:
        apply_edit(m, edit)
    return m


def param_dict(model):
    """{name: parameter} without tied duplicates (proj_out.weight is tied to model.decoder.embed_tokens.weight)."""
    return dict(model.named_parameters())


def read_edit(path):
    """An edit file -> {name: tensor}. Accepts .safetensors (preferred) or .pt. Missing/None -> {}."""
    if not path:
        return {}
    path = Path(path)
    if path.suffix == ".pt":
        return torch.load(path, map_location="cpu")
    from safetensors.torch import load_file
    return load_file(str(path))


def write_edit(path, tensors):
    from safetensors.torch import save_file
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file({k: v.detach().to("cpu", torch.float32).contiguous() for k, v in tensors.items()}, str(path))


@torch.no_grad()
def apply_edit(model, edit):
    """Shared apply-edit helper: copy {name: full new tensor} into the model's parameters (in place).
    `proj_out.weight` is accepted as an alias of the tied `model.decoder.embed_tokens.weight`.
    Returns the number of entries that changed."""
    if isinstance(edit, (str, Path)):
        edit = read_edit(edit)
    params = param_dict(model)
    changed = 0
    for name, t in edit.items():
        key = "model.decoder.embed_tokens.weight" if name == "proj_out.weight" else name
        if key not in params:
            raise KeyError(f"edit names unknown parameter {name}")
        p = params[key]
        if tuple(t.shape) != tuple(p.shape):
            raise ValueError(f"shape mismatch for {name}: {tuple(t.shape)} vs {tuple(p.shape)}")
        new = t.to(p.device, p.dtype)
        changed += int((new != p.data).sum())
        p.data.copy_(new)
    return changed


def edit_stats(edit, base_params):
    """{name: (entries changed, L2 norm of the change)} against a {name: tensor} base."""
    out = {}
    for k, v in edit.items():
        key = "model.decoder.embed_tokens.weight" if k == "proj_out.weight" else k
        d = v.float() - base_params[key].detach().float().cpu()
        out[k] = (int((d != 0).sum()), float(d.norm()))
    return out


# ----------------------------------------------------------------------------------------------- tokens
class Tok:
    """Special token ids and decoding config (HF defaults from generation_config.json)."""

    def __init__(self, model, proc):
        gc = model.generation_config
        tk = proc.tokenizer
        self.tokenizer = tk
        self.sot = gc.decoder_start_token_id
        self.eot = gc.eos_token_id
        self.lang = gc.lang_to_id["<|en|>"] if getattr(gc, "lang_to_id", None) else tk.convert_tokens_to_ids("<|en|>")
        self.transcribe = gc.task_to_id["transcribe"] if getattr(gc, "task_to_id", None) else \
            tk.convert_tokens_to_ids("<|transcribe|>")
        self.notimestamps = gc.no_timestamps_token_id
        self.nospeech = tk.convert_tokens_to_ids("<|nospeech|>")
        if self.nospeech is None or self.nospeech == tk.unk_token_id:
            self.nospeech = tk.convert_tokens_to_ids("<|nocaptions|>")
        self.prompt = [self.sot, self.lang, self.transcribe, self.notimestamps]
        self.suppress = list(gc.suppress_tokens or [])
        self.begin_suppress = list(gc.begin_suppress_tokens or [])
        self.max_new = min(MAX_POSITIONS, getattr(gc, "max_length", MAX_POSITIONS) or MAX_POSITIONS) - len(self.prompt)
        self.timestamp_begin = self.notimestamps + 1

    def text(self, ids):
        return self.tokenizer.decode([i for i in ids if i < self.eot], skip_special_tokens=True)


def features(proc, audios, device="cpu", dtype=torch.float32):
    """Log-mel input features [B, n_mels, 3000] (each clip zero-padded to 30 s, as HF does)."""
    f = proc.feature_extractor([np.asarray(a, dtype=np.float32) for a in audios], sampling_rate=SR,
                               return_tensors="pt").input_features
    return f.to(device, dtype)


def compression_ratio(text):
    """openai-whisper: len(utf-8 bytes) / len(zlib-compressed bytes)."""
    b = text.encode("utf-8")
    return len(b) / len(zlib.compress(b)) if b else 0.0


@torch.no_grad()
def encode(model, feats):
    return model.model.encoder(feats).last_hidden_state


@torch.no_grad()
def greedy_decode(model, tok, enc, max_new=None, step_hook=None):
    """Greedy decoding from encoder states enc [B, 1500, d]; matches HF's default short-form generate.
    Returns a list of dicts: tokens (generated, without EOT), nsp (no-speech prob at the SOT position, softmax of raw
    logits, as openai-whisper), avg_logprob (mean log-prob of the chosen tokens incl. EOT under the suppressed
    distribution, as openai-whisper), p_eot_first (raw prob of EOT at the first content position), hit_max.
    step_hook(step, raw_logits[B, V]) is called for every generated position (for diagnostics)."""
    B = enc.shape[0]
    dev = enc.device
    max_new = tok.max_new if max_new is None else min(max_new, tok.max_new)
    ids = torch.tensor([tok.prompt] * B, device=dev)
    sup = torch.tensor(tok.suppress, device=dev, dtype=torch.long)
    bsup = torch.tensor(tok.begin_suppress, device=dev, dtype=torch.long)
    out = model(encoder_outputs=(enc,), decoder_input_ids=ids, use_cache=True)
    raw0 = out.logits[:, 0].float()
    nsp = raw0.softmax(-1)[:, tok.nospeech].cpu()
    logits = out.logits[:, -1].float()
    past = out.past_key_values
    done = torch.zeros(B, dtype=torch.bool, device=dev)
    sum_lp = torch.zeros(B, device=dev)
    n_lp = torch.zeros(B, device=dev)
    toks = []
    p_eot_first = logits.softmax(-1)[:, tok.eot].cpu()
    for step in range(max_new):
        if step_hook is not None:
            step_hook(step, logits)
        logits = logits.clone()
        if len(sup):
            logits[:, sup] = -math.inf
        if step == 0 and len(bsup):
            logits[:, bsup] = -math.inf
        lp = logits.log_softmax(-1)
        nxt = logits.argmax(-1)
        nxt = torch.where(done, torch.full_like(nxt, tok.eot), nxt)
        chosen = lp.gather(1, nxt[:, None]).squeeze(1)
        sum_lp += torch.where(done, torch.zeros_like(chosen), chosen)
        n_lp += (~done).float()
        toks.append(nxt)
        done = done | (nxt == tok.eot)
        if bool(done.all()):
            break
        out = model(encoder_outputs=(enc,), decoder_input_ids=nxt[:, None], past_key_values=past, use_cache=True)
        logits = out.logits[:, -1].float()
        past = out.past_key_values
    T = torch.stack(toks, 1).cpu() if toks else torch.zeros(B, 0, dtype=torch.long)
    res = []
    for b in range(B):
        row = T[b].tolist()
        gen = row[:row.index(tok.eot)] if tok.eot in row else row
        res.append(dict(tokens=gen, nsp=float(nsp[b]), avg_logprob=float(sum_lp[b] / max(n_lp[b], 1)),
                        p_eot_first=float(p_eot_first[b]), hit_max=tok.eot not in row))
    return res


def transcribe(model, proc, tok, audios, batch=16, max_new=None, device="cuda", dtype=torch.float32):
    """Transcribe a list of float32 16 kHz arrays (each <= 30 s). Model must already be on `device`.
    Returns per-clip dicts with text, tokens and the openai statistics (nsp, avg_logprob, compression_ratio)."""
    res = []
    for i in range(0, len(audios), batch):
        f = features(proc, audios[i:i + batch], device, dtype)
        enc = encode(model, f)
        for r in greedy_decode(model, tok, enc, max_new=max_new):
            r["text"] = tok.text(r["tokens"])
            r["compression_ratio"] = compression_ratio(r["text"])
            res.append(r)
    return res


# ----------------------------------------------------------------------------------------------- thresholds
def threshold_keep(r, cfg):
    """openai-whisper's short-form rule, applied deterministically to one greedy result r.
    cfg keys (None = off): no_speech (skip if nsp > it AND avg_logprob <= logprob), logprob, and two optional strict
    drops used by the tuned arm: drop_logprob (drop if avg_logprob < it) and drop_compression (drop if
    compression_ratio > it). Temperature fallback is not used (it re-samples; it cannot by itself empty a transcript).
    Returns True if the transcript is kept."""
    ns, lp = cfg.get("no_speech"), cfg.get("logprob")
    if ns is not None and r["nsp"] > ns and (lp is None or r["avg_logprob"] <= lp):
        return False
    if cfg.get("drop_logprob") is not None and r["avg_logprob"] < cfg["drop_logprob"]:
        return False
    if cfg.get("drop_compression") is not None and r["compression_ratio"] > cfg["drop_compression"]:
        return False
    return True


OPENAI_DEFAULT_THRESHOLDS = dict(no_speech=0.6, logprob=-1.0, drop_logprob=None, drop_compression=None)


# ----------------------------------------------------------------------------------------------- VAD
# faster-whisper's vad_filter defaults (VadOptions): threshold 0.5, min_speech 0 ms, min_silence 2000 ms, pad 400 ms.
VAD_OPTS = dict(threshold=0.5, min_speech_duration_ms=0, min_silence_duration_ms=2000, speech_pad_ms=400)
_vad = None


def vad_model():
    global _vad
    if _vad is None:
        import warnings
        warnings.filterwarnings("ignore", message=".*torch.jit.load.*")
        from silero_vad import load_silero_vad
        _vad = load_silero_vad()
    return _vad


def vad_segments(audio, opts=None):
    """Silero VAD (CPU) speech regions as [[start, end], ...] in samples."""
    from silero_vad import get_speech_timestamps
    o = dict(VAD_OPTS, **(opts or {}))
    ts = get_speech_timestamps(torch.from_numpy(np.array(audio, dtype=np.float32)), vad_model(),
                               sampling_rate=SR, **o)
    return [[int(t["start"]), int(t["end"])] for t in ts]


def vad_collect(audio, segs):
    """faster-whisper style: concatenate the detected speech regions (None if there are none)."""
    if not segs:
        return None
    return np.concatenate([np.asarray(audio[s:e], dtype=np.float32) for s, e in segs])


# ----------------------------------------------------------------------------------------------- metrics
_norm = None


def normalizer(model_name=DEFAULT_MODEL):
    """Whisper's EnglishTextNormalizer (with the model's normalizer.json spelling map)."""
    global _norm
    if _norm is None:
        from huggingface_hub import hf_hub_download
        from transformers.models.whisper.english_normalizer import EnglishTextNormalizer
        try:
            spell = json.loads(Path(hf_hub_download(DEFAULT_MODEL, "normalizer.json")).read_text(encoding="utf-8"))
        except Exception:
            spell = {}
        _norm = EnglishTextNormalizer(spell)
    return _norm


def norm(text):
    return " ".join(normalizer()(text or "").split())


# conventional caption tags; an output made only of these words (<= 4 words, e.g. "Music", "Applause",
# "Upbeat music") is a sound tag, not invented speech. Bracketed / parenthesised tags already normalize to empty.
SOUND_TAG_WORDS = {"music", "applause", "laughter", "laughing", "laughs", "silence", "noise", "static", "inaudible",
                   "cheering", "instrumental", "upbeat", "playing", "background", "soft", "dramatic", "gentle",
                   "sighs", "coughs", "coughing", "clapping"}


def is_sound_tag(normalized):
    w = normalized.split()
    return 0 < len(w) <= 4 and all(x in SOUND_TAG_WORDS for x in w)


def hallucination(rows, key="text"):
    """Per category and overall: rate = share of non-empty normalized transcripts (primary); rate_notag = the same
    but not counting outputs that are only a sound tag like "Music" (secondary). Rows need 'category'."""
    out = {}
    cats = sorted(set(r["category"] for r in rows))
    nn = [norm(r[key]) for r in rows]
    for c in ["overall"] + cats:
        idx = [i for i, r in enumerate(rows) if c == "overall" or r["category"] == c]
        if idx:
            out[c] = dict(rate=round(float(np.mean([bool(nn[i]) for i in idx])), 4),
                          rate_notag=round(float(np.mean([bool(nn[i]) and not is_sound_tag(nn[i]) for i in idx])), 4),
                          n=len(idx))
    return out


def word_stats(refs, hyps):
    """jiwer alignment on normalized text. Returns corpus WER (%), insertion rate (insertions per reference word, %),
    share of utterances with any insertion, deletion/substitution rates, dropped = share of utterances whose
    transcript is empty after normalization (fully dropped), and per-utterance counts."""
    import jiwer
    per, S = [], dict(sub=0, dele=0, ins=0, hits=0, nref=0)
    for r, h in zip(refs, hyps):
        rn, hn = norm(r), norm(h)
        if not rn:
            continue
        o = jiwer.process_words(rn, hn if hn else "")
        d = dict(sub=o.substitutions, dele=o.deletions, ins=o.insertions, hits=o.hits, nref=len(rn.split()),
                 dropped=not hn)
        per.append(d)
        for k in S:
            S[k] += d[k]
    n = max(S["nref"], 1)
    return dict(wer=round(100 * (S["sub"] + S["dele"] + S["ins"]) / n, 3),
                ins_rate=round(100 * S["ins"] / n, 3), del_rate=round(100 * S["dele"] / n, 3),
                sub_rate=round(100 * S["sub"] / n, 3),
                any_ins=round(float(np.mean([d["ins"] > 0 for d in per])) if per else 0.0, 4),
                dropped=round(float(np.mean([d["dropped"] for d in per])) if per else 0.0, 4),
                n=len(per), nref=S["nref"], per=per)


def _by(rows, field, key):
    out = {}
    for v in sorted(set(r[field] for r in rows), key=str):
        rr = [r for r in rows if r[field] == v]
        w = word_stats([r["ref"] for r in rr], [r[key] for r in rr])
        w.pop("per")
        out[str(v)] = w
    return out


def suite_metrics(suite, rows, key="text"):
    """Metrics for one suite's result rows (each has id, category, text, and ref for speech suites)."""
    if suite in ("nonspeech", "practice_nonspeech"):
        return dict(hallucination=hallucination(rows, key))
    ws = word_stats([r["ref"] for r in rows], [r[key] for r in rows])
    ws.pop("per")
    out = dict(words=ws)
    first = rows[0] if rows else {}
    if "pad" in first:
        out["by_pad"] = _by(rows, "pad", key)
    if "bg" in first:
        out["by_bg"] = _by(rows, "bg", key)
        out["by_snr"] = _by(rows, "snr", key)
    return out
