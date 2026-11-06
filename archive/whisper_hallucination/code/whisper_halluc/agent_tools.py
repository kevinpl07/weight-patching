"""Tools for a weight-editing agent session on Whisper (openai/whisper-small by default; HF transformers).

The session folder (AGENT_SESSION, default: current directory) contains PROMPT.md and work/ (notes, tool log, edit).

Usage from an agent script:
    import sys; sys.path.insert(0, "<repo>/whisper_halluc"); import agent_tools as T
    T.info()                         # architecture, special tokens, decoding rules, budgets, edit scope, practice data
    W = T.base_weights()             # {name: CPU float32 tensor}, read-only copy of all weights
    P = T.practice()                 # practice clips: ids, audio (16 kHz float32), kind, category, ref text, ...
    T.spectrogram(["pns_env_003", 0], "work/spec.png")   # save log-mel images of clips to look at
    out = T.run(["pns_sil_000", "ps_251-137823-0025"], edited=True, cross_attn=True, hidden=True)
    m = T.metrics(edited=True)       # practice diagnostics: hallucination per category, WER, insertions
    T.save_edit({"model.decoder.layers.3.fc2.weight": new}, note="..."); T.current_edit(); T.budget()

Forward passes are counted per call (budget AGENT_PASSES, default 30): run() costs 1 pass (at most 64 clips),
metrics() costs 2 (all 300 practice clips). Every call takes the repo's GPU lock and is logged to work/tool_log.jsonl. There are no gradient
tools, and the rules forbid computing gradients yourself.
"""
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from gpu_lock import gpu  # noqa: E402

SESSION = Path(os.environ.get("AGENT_SESSION", Path.cwd()))
WORK = SESSION / "work"
EDIT = WORK / "edited" / "edit.safetensors"
MODEL = os.environ.get("AGENT_MODEL", C.DEFAULT_MODEL)
MAX_PASSES = int(os.environ.get("AGENT_PASSES", 30))
ALLOW_ENCODER = os.environ.get("AGENT_ALLOW_ENCODER", "0") == "1"
MAX_CLIPS = 64
METRICS_PASSES = 2
MAX_RETURN_BYTES = 4e9
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TIED = "model.decoder.embed_tokens.weight"

_base = None
_proc = None
_models = {}
_practice = None
_max_changed = None


# ----------------------------------------------------------------------------------------------- bookkeeping
def _log(kind, **kw):
    WORK.mkdir(parents=True, exist_ok=True)
    with open(WORK / "tool_log.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(t=time.strftime("%H:%M:%S"), kind=kind, **kw)) + "\n")


def _spend(n, kind):
    path = WORK / "budget.json"
    WORK.mkdir(parents=True, exist_ok=True)
    used = json.loads(path.read_text())["used"] if path.exists() else 0
    if used + n > MAX_PASSES:
        raise RuntimeError(f"forward-pass budget exhausted ({used}+{n} > {MAX_PASSES})")
    path.write_text(json.dumps(dict(used=used + n, max=MAX_PASSES)))
    print(f"[budget] passes {used + n}/{MAX_PASSES} ({kind})")


def budget():
    """Print and return the passes used so far. Free."""
    path = WORK / "budget.json"
    used = json.loads(path.read_text())["used"] if path.exists() else 0
    print(f"passes used {used}/{MAX_PASSES}; edit budget {max_changed():,} entries; current edit: "
          f"{'yes' if EDIT.exists() else 'none'}")
    return used


# ----------------------------------------------------------------------------------------------- weights
def _processor():
    global _proc
    if _proc is None:
        _proc = C.load_processor(MODEL)
    return _proc


def tokenizer():
    """The HF Whisper tokenizer (encode/decode/convert_ids_to_tokens). Free."""
    return _processor().tokenizer


def base_weights():
    """All weights as CPU float32 tensors {name: tensor} (encoder included, for reading). Free. Copy before modifying.
    proj_out (the output projection) is tied to model.decoder.embed_tokens.weight and is not listed separately."""
    global _base
    if _base is None:
        m = C.load_model(MODEL)
        _base = {k: v.detach().float().clone() for k, v in C.param_dict(m).items()}
        del m
    return _base


def editable():
    """Names of the tensors an edit may change. Free."""
    return [k for k in base_weights() if k.startswith("model.decoder.") or (ALLOW_ENCODER and k.startswith("model.encoder."))]


def max_changed():
    """Edit budget: 1% of the decoder's entries (absolute count), unless AGENT_MAX_CHANGED is set."""
    global _max_changed
    if _max_changed is None:
        env = os.environ.get("AGENT_MAX_CHANGED")
        if env:
            _max_changed = int(env)
        else:
            n = sum(v.numel() for k, v in base_weights().items() if k.startswith("model.decoder."))
            _max_changed = int(0.01 * n)
    return _max_changed


def current_edit():
    """The current edit as {name: full tensor}, or {} if none. Free."""
    return C.read_edit(EDIT) if EDIT.exists() else {}


def save_edit(tensors, note=""):
    """Validate and save {param_name: full new tensor} as the current edit (replaces the previous one). Free.
    Rules: names must be in editable() (proj_out.weight is accepted as the tied embedding); shapes must match; values
    must be finite; at most max_changed() entries may differ from the base weights in total."""
    base = base_weights()
    scope = set(editable())
    changed, clean, norms = 0, {}, {}
    for name, t in tensors.items():
        key = TIED if name == "proj_out.weight" else name
        if key not in scope:
            raise ValueError(f"unknown or out-of-scope tensor {name}")
        t = torch.as_tensor(t).detach().to("cpu", torch.float32).contiguous()
        if t.shape != base[key].shape:
            raise ValueError(f"shape mismatch for {name}: {tuple(t.shape)} vs {tuple(base[key].shape)}")
        if not torch.isfinite(t).all():
            raise ValueError(f"non-finite values in {name}")
        n = int((t != base[key]).sum())
        if n:
            changed += n
            clean[key] = t
            norms[key] = round((t - base[key]).norm().item(), 4)
    if changed > max_changed():
        raise ValueError(f"edit changes {changed:,} entries; the limit is {max_changed():,}")
    C.write_edit(EDIT, clean)
    _log("save_edit", tensors=sorted(clean), changed=changed, delta_norms=norms, note=note)
    print(f"saved edit: {len(clean)} tensors, {changed:,} entries changed, delta norms {norms}")
    return changed


def _get_model(edited):
    key = ("edited", EDIT.stat().st_mtime) if (edited and EDIT.exists()) else ("base",)
    if key not in _models:
        for k in [k for k in _models if k[0] == key[0]]:
            del _models[k]
        _models[key] = C.load_model(MODEL, edit=current_edit() if key[0] == "edited" else None, attn="eager")
    return _models[key]


def _tok(model):
    return C.Tok(model, _processor())


# ----------------------------------------------------------------------------------------------- data
def practice():
    """The practice clips (all of them; free). Returns a dict of parallel lists:
      ids       e.g. 'pns_env_003' (non-speech), 'ps_<utt>' (clean speech), 'ppad_<utt>' (speech with padding)
      audio     float32 numpy arrays, 16 kHz mono, each <= 30 s (Whisper zero-pads every input to 30 s)
      kind      'nonspeech' (100 clips), 'speech' (100 clean LibriSpeech dev-clean utterances),
                'padded' (50 other dev-clean utterances with 5-10 s of silence / noise / music before and after) or
                'background' (50 other dev-clean utterances with an environmental sound or music mixed under the whole
                utterance at a speech-to-background ratio of 0, 5 or 10 dB)
      category  nonspeech: 'silence' | 'noise' | 'environmental' | 'music'; speech and padded: 'speech'
      sub       finer source label (noise colour and level, ESC-50 class, music genre)
      ref       reference transcript for speech and padded clips (None for non-speech)
      pad       padding type for padded clips ('silence' | 'noise' | 'music'), else None
      bg, snr   background type ('env' | 'music') and SNR in dB for background clips, else None
      seconds   duration"""
    global _practice
    if _practice is None:
        d = {k: [] for k in ("ids", "audio", "kind", "category", "sub", "ref", "pad", "bg", "snr", "seconds")}
        for suite, kind in (("practice_nonspeech", "nonspeech"), ("practice_speech", "speech"),
                            ("practice_padded", "padded"), ("practice_bg", "background")):
            man, aud = C.load_suite(suite)
            for m, x in zip(man, aud):
                d["ids"].append(m["id"])
                d["audio"].append(np.array(x, dtype=np.float32))
                d["kind"].append(kind)
                d["category"].append(m["category"])
                d["sub"].append(m.get("sub"))
                d["ref"].append(m.get("ref"))
                d["pad"].append(m.get("pad"))
                d["bg"].append(m.get("bg"))
                d["snr"].append(m.get("snr"))
                d["seconds"].append(m["seconds"])
        _practice = d
    return _practice


def _resolve(clips):
    """clips: list of practice ids (str), practice indices (int) or float32 16 kHz arrays (<= 30 s)."""
    P = practice()
    pos = {k: i for i, k in enumerate(P["ids"])}
    ids, audio = [], []
    for c in clips:
        if isinstance(c, str):
            ids.append(c)
            audio.append(P["audio"][pos[c]])
        elif isinstance(c, (int, np.integer)):
            ids.append(P["ids"][int(c)])
            audio.append(P["audio"][int(c)])
        else:
            x = np.asarray(c.detach().cpu().numpy() if torch.is_tensor(c) else c, dtype=np.float32).reshape(-1)
            if len(x) > 30 * C.SR:
                raise ValueError("arrays must be at most 30 s at 16 kHz")
            ids.append(f"array_{len(ids)}")
            audio.append(x)
    return ids, audio


def spectrogram(clips, path, ncol=2):
    """Save the Whisper log-mel input (80 x 3000 frames = 30 s, after zero-padding) of up to 16 clips as one PNG,
    titled with id / category / reference. Open the PNG with your file-reading tool to look at it. Free."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ids, audio = _resolve(list(clips)[:16])
    f = C.features(_processor(), audio).numpy()
    P = practice()
    pos = {k: i for i, k in enumerate(P["ids"])}
    nrow = math.ceil(len(ids) / ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(7 * ncol, 2.2 * nrow), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for i, (cid, x) in enumerate(zip(ids, f)):
        ax = axes.flat[i]
        ax.imshow(x, origin="lower", aspect="auto", cmap="magma", extent=[0, 30, 0, x.shape[0]])
        j = pos.get(cid)
        lab = cid if j is None else (f"{cid} [{P['category'][j]}/{P['sub'][j] or P['bg'][j] or ''}"
                                     f"{'' if P['snr'][j] is None else ' ' + str(P['snr'][j]) + ' dB'}] "
                                     f"{(P['ref'][j] or '')[:60]}")
        ax.set_title(lab, fontsize=7)
        ax.axis("on")
        ax.set_xlabel("s", fontsize=6)
        ax.tick_params(labelsize=6)
    fig.tight_layout()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=80)
    plt.close(fig)
    return str(path)


# ----------------------------------------------------------------------------------------------- forward passes
@torch.no_grad()
def run(clips, edited=True, forced=None, max_new_tokens=None, topk=10, full_logits=False, encoder=False,
        self_attn=False, cross_attn=False, hidden=False, modules=(), attn_layers=None):
    """Transcribe clips with the deployed decoding, then re-run the decoder teacher-forced on the result to record
    internals. Costs 1 pass; at most 64 clips.
      clips        practice ids, practice indices, or your own float32 16 kHz arrays (<= 30 s)
      edited       use the current edit (if one is saved) instead of the base weights
      forced       optional list (one per clip) of content-token id lists to teacher-force instead of generating
                   (no prompt, no end-of-text; e.g. tokenizer().encode(" Thank you.", add_special_tokens=False))
      max_new_tokens  cap on generated tokens (default: the deployed limit, 444)
    Returns a list (one dict per clip) with:
      id, text, norm_text (Whisper's English normalizer; empty = counted as no hallucination), tokens (generated
      content ids, no end-of-text), seq (the decoder input: 4 prompt tokens + tokens + end-of-text), seq_str
      nsp          probability of <|nospeech|> at the <|startoftranscript|> position (raw softmax)
      avg_logprob  mean log-prob of the chosen tokens incl. end-of-text under the deployed (suppressed) distribution
      p_eot_first  raw probability of end-of-text at the first content position (the deployed decoding suppresses
                   it there, see info())
      topk_ids, topk_logprobs   [L, topk] raw log-softmax at every position of seq (position i predicts seq[i+1])
      logits       [L, V] raw logits, float16 (only if full_logits)
      encoder      [1500, d_model] encoder output, float16 (if encoder)
      self_attn    [n_layers, n_heads, L, L] decoder self-attention probabilities, float16 (if self_attn)
      cross_attn   [n_layers, n_heads, L, 1500] decoder cross-attention probabilities, float16 (if cross_attn);
                   1500 encoder frames = 30 s, 20 ms each. attn_layers=[...] keeps only those layers.
      hidden       [n_layers + 1, L, d_model] decoder residual stream (embeddings, then each layer's output) (if hidden)
      modules      {name: output [L, ...]} for any decoder submodule names, e.g. "model.decoder.layers.5.self_attn",
                   "model.decoder.layers.5.fc1", "model.decoder.layers.5.encoder_attn.out_proj" (attention modules
                   return their output projection result)"""
    ids, audio = _resolve(list(clips))
    if len(ids) > MAX_CLIPS:
        raise ValueError(f"{len(ids)} clips > {MAX_CLIPS} per call")
    if forced is not None and len(forced) != len(ids):
        raise ValueError("forced needs one token list per clip")
    _spend(1, "run")
    model = _get_model(edited)
    tok = _tok(model)
    cfg = model.config
    named = dict(model.named_modules())
    for n in modules:
        if n not in named:
            raise ValueError(f"unknown module {n}")
    out = []
    with gpu(f"whisper agent {SESSION.name} run"):
        C.gpu_setup()
        model.to(DEVICE)
        try:
            feats = C.features(_processor(), audio, DEVICE)
            enc_all = torch.cat([C.encode(model, feats[i:i + 16]) for i in range(0, len(ids), 16)])
            gen = []
            if forced is None:
                for i in range(0, len(ids), 16):
                    gen += C.greedy_decode(model, tok, enc_all[i:i + 16], max_new=max_new_tokens)
            else:
                for i in range(0, len(ids), 16):
                    g = C.greedy_decode(model, tok, enc_all[i:i + 16], max_new=1)  # for nsp / p_eot_first
                    gen += g
                for g, f in zip(gen, forced):
                    g["tokens"] = [int(t) for t in f]
                    g["avg_logprob"] = None
            L_max = max(len(tok.prompt) + len(g["tokens"]) + 1 for g in gen)
            n_l, n_h = cfg.decoder_layers, cfg.decoder_attention_heads
            nl_keep = len(attn_layers) if attn_layers is not None else n_l
            est = len(ids) * L_max * (2 * cfg.vocab_size * full_logits + 2 * cfg.d_model * 1500 * 0 +
                                      2 * nl_keep * n_h * (L_max * self_attn + 1500 * cross_attn) +
                                      4 * (n_l + 1) * cfg.d_model * hidden) + len(ids) * 1500 * cfg.d_model * 2 * encoder
            if est > MAX_RETURN_BYTES:
                raise ValueError(f"requested internals would take ~{est / 1e9:.1f} GB; use fewer clips, attn_layers, "
                                 f"or max_new_tokens")
            for i in range(0, len(ids), 8):
                gb = gen[i:i + 8]
                seqs = [tok.prompt + g["tokens"] + [tok.eot] for g in gb]
                Lb = max(len(s) for s in seqs)
                x = torch.tensor([s + [tok.eot] * (Lb - len(s)) for s in seqs], device=DEVICE)
                cap, hooks = {}, []
                for n in modules:
                    hooks.append(named[n].register_forward_hook(
                        lambda mod, inp, o, n=n: cap.__setitem__(n, (o[0] if isinstance(o, tuple) else o).detach())))
                try:
                    o = model(encoder_outputs=(enc_all[i:i + 8],), decoder_input_ids=x, use_cache=False,
                              output_attentions=bool(self_attn or cross_attn), output_hidden_states=bool(hidden))
                finally:
                    for h in hooks:
                        h.remove()
                lsm = o.logits.float().log_softmax(-1)
                tv, ti = lsm.topk(topk, -1)
                keep = list(range(n_l)) if attn_layers is None else list(attn_layers)
                for b, (g, s) in enumerate(zip(gb, seqs)):
                    L = len(s)
                    text = tok.text(g["tokens"])
                    r = dict(id=ids[i + b], text=text, norm_text=C.norm(text), tokens=g["tokens"], seq=s,
                             seq_str=tokenizer().convert_ids_to_tokens(s), nsp=g["nsp"],
                             avg_logprob=g["avg_logprob"], p_eot_first=g["p_eot_first"],
                             compression_ratio=C.compression_ratio(text),
                             topk_ids=ti[b, :L].cpu(), topk_logprobs=tv[b, :L].cpu())
                    if full_logits:
                        r["logits"] = o.logits[b, :L].half().cpu()
                    if encoder:
                        r["encoder"] = enc_all[i + b].half().cpu()
                    if self_attn:
                        r["self_attn"] = torch.stack([o.decoder_attentions[l][b, :, :L, :L] for l in keep]).half().cpu()
                    if cross_attn:
                        r["cross_attn"] = torch.stack([o.cross_attentions[l][b, :, :L] for l in keep]).half().cpu()
                    if hidden:
                        r["hidden"] = torch.stack([h[b, :L] for h in o.decoder_hidden_states]).float().cpu()
                    if modules:
                        r["modules"] = {n: v[b, :L].float().cpu() for n, v in cap.items()}
                    out.append(r)
        finally:
            model.cpu()
            torch.cuda.empty_cache()
    _log("run", n=len(ids), edited=bool(edited and EDIT.exists()), forced=forced is not None,
         extras=[k for k, v in dict(full_logits=full_logits, encoder=encoder, self_attn=self_attn,
                                     cross_attn=cross_attn, hidden=hidden).items() if v] + list(modules),
         ids=ids[:64])
    return out


def metrics(edited=True):
    """Practice-set diagnostics with the deployed decoding (costs 2 passes; ~1-2 min):
      halluc          share of the 100 non-speech clips whose normalized transcript is non-empty, overall and
                      per category (silence / noise / environmental / music); halluc_notag also ignores outputs that
                      are only a sound tag such as "Music"
      wer             word error rate (%) on the 100 clean speech clips (normalized, jiwer)
      bg_wer, bg_dropped   WER on the 50 speech-in-background clips, and the share of them whose transcript is empty
      padded_ins      inserted words per reference word (%) on the 50 padded clips; padded_any_ins = share of padded
                      clips with any inserted word; padded_wer
      transcripts     {id: text} for every practice clip"""
    _spend(METRICS_PASSES, "metrics")
    model = _get_model(edited)
    tok = _tok(model)
    P = practice()
    with gpu(f"whisper agent {SESSION.name} metrics"):
        C.gpu_setup()
        model.to(DEVICE)
        try:
            res = C.transcribe(model, _processor(), tok, P["audio"], batch=16, device=DEVICE)
        finally:
            model.cpu()
            torch.cuda.empty_cache()
    text = {i: r["text"] for i, r in zip(P["ids"], res)}
    ns = [dict(category=c, text=text[i]) for i, k, c in zip(P["ids"], P["kind"], P["category"]) if k == "nonspeech"]
    h = C.hallucination(ns)
    sp = [(r, text[i]) for i, k, r in zip(P["ids"], P["kind"], P["ref"]) if k == "speech"]
    pp = [(r, text[i]) for i, k, r in zip(P["ids"], P["kind"], P["ref"]) if k == "padded"]
    bb = [(r, text[i]) for i, k, r in zip(P["ids"], P["kind"], P["ref"]) if k == "background"]
    ws = C.word_stats(*zip(*sp))
    wp = C.word_stats(*zip(*pp))
    wb = C.word_stats(*zip(*bb))
    m = dict(halluc=h["overall"]["rate"], halluc_notag=h["overall"]["rate_notag"],
             halluc_by_cat={k: v["rate"] for k, v in h.items() if k != "overall"},
             wer=ws["wer"], bg_wer=wb["wer"], bg_dropped=wb["dropped"],
             padded_ins=wp["ins_rate"], padded_any_ins=wp["any_ins"], padded_wer=wp["wer"])
    _log("metrics", edited=bool(edited and EDIT.exists()), **m)
    print(json.dumps(m))
    m["transcripts"] = text
    return m


def info():
    """Print and return a summary of the model, decoding, budgets, edit scope and practice data. Free."""
    W = base_weights()
    m = _get_model(False)
    cfg = m.config
    tok = _tok(m)
    P = practice()
    cnt = {}
    for k, c in zip(P["kind"], P["category"]):
        cnt[f"{k}/{c}"] = cnt.get(f"{k}/{c}", 0) + 1
    dec = sum(v.numel() for k, v in W.items() if k.startswith("model.decoder."))
    s = dict(model=f"{MODEL} (HF transformers WhisperForConditionalGeneration, float32, eval mode)",
             params=sum(v.numel() for v in W.values()), decoder_params=dec,
             d_model=cfg.d_model, encoder_layers=cfg.encoder_layers, decoder_layers=cfg.decoder_layers,
             decoder_heads=cfg.decoder_attention_heads, d_head=cfg.d_model // cfg.decoder_attention_heads,
             ffn_dim=cfg.decoder_ffn_dim, vocab=cfg.vocab_size,
             layout=("pre-LN transformer; each decoder layer: self_attn_layer_norm -> self_attn (q/k/v/out_proj, causal) "
                     "-> encoder_attn_layer_norm -> encoder_attn (cross-attention over 1500 encoder frames) -> "
                     "final_layer_norm -> fc1 -> GELU -> fc2, each with a residual add; then model.decoder.layer_norm "
                     "and proj_out, which is tied to model.decoder.embed_tokens.weight (editing that tensor changes "
                     "both the input embedding and the output projection)."),
             special_tokens=dict(sot=tok.sot, en=tok.lang, transcribe=tok.transcribe, notimestamps=tok.notimestamps,
                                 nospeech=tok.nospeech, eot=tok.eot, timestamp_begin=tok.timestamp_begin),
             decoding=dict(prompt=tok.prompt, method="greedy", max_new_tokens=tok.max_new,
                           suppressed_every_step=len(tok.suppress), suppressed_at_first_step=tok.begin_suppress,
                           note=("the deployed decoding forbids end-of-text and a bare space as the first content "
                                 "token, so the shortest possible output is one token; a transcript counts as empty "
                                 "(no hallucination) if Whisper's English normalizer reduces it to nothing, e.g. "
                                 "punctuation only, symbols, or text in [brackets] / (parentheses)")),
             edit_budget_entries=max_changed(), edit_scope=("decoder tensors (model.decoder.*)" +
                                                            (" and encoder tensors" if ALLOW_ENCODER else "")),
             forward_budget_passes=MAX_PASSES, max_clips_per_run=MAX_CLIPS, metrics_cost_passes=METRICS_PASSES,
             practice=dict(clips=len(P["ids"]), by_kind_category=cnt))
    print(json.dumps(s, indent=1))
    return s
