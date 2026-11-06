"""Whisper writes text on silence, noise and music (openai/whisper-small; also runs on whisper-large-v3-turbo).

Whisper is an encoder-decoder, so this task implements its own commands (patchkit's generic ones assume chat LLMs)
but writes the same output contract (docs/ADDING_A_TASK.md), so patchkit.report and the site read it unchanged.

    failure  hallucination = % of 300 held-out non-speech clips (silence, noise, ESC-50 sounds, instrumental music)
             whose transcript is non-empty after Whisper's EnglishTextNormalizer (" ." or "[Music]" count as empty)
    kept     speech_kept   = % of held-out guard utterances (padded 300 + clean 500 + speech-over-background 300 +
             LibriSpeech test-other 300 = 1400) whose normalized transcript is identical to the base model's.
             Any change counts against it, including improvements; WER deltas in `metrics` show the direction.

Arms: base | thresholds (openai default no-speech/log-prob rule, 0.6 / -1.0, on one greedy pass) | thresholds_tuned
(same rule, threshold tuned on practice data in the original run) | vad (Silero, faster-whisper defaults) | gate (one
neuron in the last decoder layer reads the no-speech direction at the first content position and writes " .") |
random (same neuron, random vectors of the same norms) | lora (LoRA r=16 on the decoder, trained on the same practice
clips: non-speech -> " .", speech -> the base model's own transcript).

    python -m tasks.whisper_silence base                 # GPU ~1 min: base decode + activations on 300 practice clips
    python -m tasks.whisper_silence vad                  # CPU: Silero VAD segments for the held-out suites (cached)
    python -m tasks.whisper_silence fit [--n N --seed S] # CPU: gate + matched random edit from the practice cache
    python -m tasks.whisper_silence lora [--n N --seed S]   # GPU ~2 min
    python -m tasks.whisper_silence eval --arms base,thresholds,thresholds_tuned,vad,gate,random,lora
    python -m tasks.whisper_silence sweep --ns 8,16,32,64,all --seeds 0,1,2 --methods gate,lora
    python -m tasks.whisper_silence audio --ids ns_sil_000,...  # export <=8 s WAV excerpts to site/audio/
    any command: --model openai/whisper-large-v3-turbo  (results go to results/whisper_silence__whisper-large-v3-turbo)

Every GPU command should run under the repo's GPU lock (see NOTES). `eval` must run `base` first (the kept metric
and the threshold arms read the base transcripts). Data: $WHISPER_SILENCE_DATA (default ../data_audio), built by
tasks/_whisper_prepare_data.py.
"""
import argparse
import json
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

WP = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(WP))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import _whisper_core as W  # noqa: E402
from patchkit.task import Task, register, subset  # noqa: E402

GAIN, TOPK = 4.0, 5                                  # gate recipe constants (fixed in the original run)
LORA = dict(r=16, alpha=32, dropout=0.05, lr=2e-4, epochs=3, min_steps=30, batch=4)   # = patchkit's LoRA recipe
LORA_TARGETS = r"model\.decoder\.layers\.\d+\.((self_attn|encoder_attn)\.(q_proj|k_proj|v_proj|out_proj)|fc1|fc2)"
SWEEP_SUBSET = dict(nonspeech=slice(None), padded=slice(0, None, 2), wer=slice(0, 150), bg=slice(0, None, 2))
KL_SUITE, KL_N = "wer", 100
SUITE_NAME = dict(wer="clean", padded="padded", bg="bg", other="other")


@register
class WhisperSilence(Task):
    name = "whisper_silence"
    model = "openai/whisper-small"
    description = "Whisper transcribes words on silence, noise and music"
    headline = ("hallucination", "speech_kept")
    headline_labels = ("non-speech clips given a transcript", "speech transcripts unchanged vs base")

    def build_items(self, ctx):
        raise NotImplementedError("audio task: use python -m tasks.whisper_silence <cmd>")

    def positive(self, item):
        return item["suite"] == "practice_nonspeech"

    def score(self, rows, base_rows=None):
        return score(rows, base_rows)


# ======================================================================== context
class Ctx:
    def __init__(self, a):
        self.a = a
        self.model_name = a.model or WhisperSilence.model
        self.name = WhisperSilence.name + ("" if not a.model or a.model == WhisperSilence.model
                                           else "__" + a.model.split("/")[-1])
        self.runs = WP / "runs" / self.name
        self.res = WP / "results" / self.name
        self.runs.mkdir(parents=True, exist_ok=True)
        self.res.mkdir(parents=True, exist_ok=True)
        self._pm = None

    @property
    def pm(self):
        if self._pm is None:
            t = time.time()
            self._pm = W.load(self.model_name)
            print(f"loaded {self.model_name} in {time.time() - t:.0f}s", flush=True)
        return self._pm

    def fresh_model(self):
        from transformers import WhisperForConditionalGeneration
        m = WhisperForConditionalGeneration.from_pretrained(self.model_name, dtype=torch.float32,
                                                            attn_implementation="sdpa")
        return m.eval()

    def practice_cache(self):
        if getattr(self, "_cache", None) is None:
            self._cache = torch.load(self.runs / "practice_base.pt", weights_only=False)
        return self._cache

    def practice_items(self, n=None, seed=0):
        cache = self.practice_cache()
        items = [dict(id=r["id"], suite=s) for s in W.PRACTICE_SUITES for r in cache["rows"][s]]
        return subset(items, n, seed, lambda x: x["suite"] == "practice_nonspeech"), cache

    def base_rows(self):
        d = self.runs / "eval" / "base"
        if not d.exists():
            raise SystemExit("run `eval --arms base` first")
        return {s: read_jsonl(d / f"{s}.jsonl") for s in W.HELDOUT_SUITES if (d / f"{s}.jsonl").exists()}


def read_jsonl(p):
    return [json.loads(l) for l in Path(p).read_text(encoding="utf-8").splitlines() if l.strip()]


def write_jsonl(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")


def tagname(a):
    return a.tag or (f"n{a.n}_s{a.seed}" if a.n else "")


# ======================================================================== scoring
def score(rows, base_rows=None):
    """rows: {suite: [row with id, category, text, ref]}. base_rows: same for the base arm (for speech_kept)."""
    m = {}
    if "nonspeech" in rows:
        ns = rows["nonspeech"]
        nn = [W.norm(r["text"]) for r in ns]
        m["hallucination"] = round(100 * float(np.mean([bool(x) for x in nn])), 2)
        m["hallucination_notag"] = round(100 * float(np.mean([bool(x) and not W.is_sound_tag(x) for x in nn])), 2)
        for c in W.CATEGORIES:
            idx = [i for i, r in enumerate(ns) if r["category"] == c]
            m[f"halluc_{c}"] = round(100 * float(np.mean([bool(nn[i]) for i in idx])), 2)
        m["n_nonspeech"] = len(ns)
    same, tot = 0, 0
    for s in W.GUARD_SUITES:
        if s not in rows:
            continue
        rr, k = rows[s], SUITE_NAME[s]
        ws = W.word_stats([r["ref"] for r in rr], [r["text"] for r in rr])
        m[f"{k}_wer"], m[f"{k}_dropped"] = ws["wer"], ws["dropped"]
        if s == "padded":
            m["padded_ins_per_word"], m["padded_any_ins"] = ws["ins"], ws["any_ins"]
        if s == "bg":
            for snr in (0, 5, 10):
                sub = [r for r in rr if r.get("snr") == snr]
                if sub:
                    m[f"bg_wer_snr{snr}"] = W.word_stats([r["ref"] for r in sub], [r["text"] for r in sub])["wer"]
        if base_rows is not None:
            b = {r["id"]: W.norm(r["text"]) for r in base_rows[s]}
            eq = [W.norm(r["text"]) == b[r["id"]] for r in rr]
            m[f"kept_{k}"] = round(100 * float(np.mean(eq)), 2)
            same, tot = same + sum(eq), tot + len(eq)
        m[f"n_{k}"] = len(rr)
    if tot:
        m["speech_kept"] = round(100 * same / tot, 2)
    return m


def result(c, arm, metrics, kl, meta, split="heldout", n=None, eval_s=None):
    meta = meta or {}
    return dict(task=c.name, model=c.model_name, arm=arm, split=split, n=n,
                headline=dict(hallucination=metrics.get("hallucination"), speech_kept=metrics.get("speech_kept")),
                metrics=metrics,
                guards=dict(mmlu200=None, chat_kl=kl,
                            chat_kl_note=f"decoder KL(base||arm) per token, teacher-forced on base transcripts of "
                                         f"{KL_N} held-out clean utterances (no chat for an ASR model)"),
                cost=dict(entries_changed=meta.get("entries_changed"), trainable_params=meta.get("trainable_params"),
                          fit_seconds=meta.get("fit_seconds"), n_practice_items=meta.get("n_items")),
                fit_meta=meta, eval_seconds=eval_s)


# ======================================================================== decoding helpers
def run_rows(model, proc, tok, suite, man, audio, batch, vad=None):
    """Transcribe (on DEV) -> rows with id, suite, category, ref..., raw_text, text, statistics."""
    inputs, extra = [], []
    for m, x in zip(man, audio):
        e = {}
        if vad is not None:
            sg = vad[m["id"]]
            x = W.vad_collect(x, sg)
            e = dict(vad_speech_s=round(sum(b - a for a, b in sg) / W.SR, 2), vad_skipped=x is None)
        inputs.append(x)
        extra.append(e)
    out = W.transcribe(model, proc, tok, inputs, batch=batch)
    rows = []
    for m, r, e in zip(man, out, extra):
        row = {k: m[k] for k in ("id", "category", "sub", "ref", "pad", "bg", "snr") if k in m}
        row["suite"] = suite
        row.update(e)
        if r is None:
            row.update(raw_text="", text="", n_tokens=0)
        else:
            row.update(raw_text=r["text"], text=r["text"], n_tokens=len(r["tokens"]), nsp=round(r["nsp"], 5),
                       avg_logprob=round(r["avg_logprob"], 4), compression_ratio=round(r["compression_ratio"], 3),
                       hit_max=r["hit_max"], tokens=r["tokens"])
        rows.append(row)
    return rows


def load_suites(suites, sub=None):
    out = {}
    for s in suites:
        man, aud = W.load_suite(s)
        if sub and s in sub:
            man, aud = man[sub[s]], aud[sub[s]]
        out[s] = (man, aud)
    return out


@torch.no_grad()
def speech_kl(model, base_model, proc, tok, base_rows, batch=16):
    """Mean per-token KL(base || arm), teacher-forced on the base model's transcripts of KL_N clean utterances."""
    rows = base_rows[KL_SUITE][:KL_N]
    man, aud = W.load_suite(KL_SUITE)
    amap = {m["id"]: a for m, a in zip(man, aud)}
    vals = []
    for i in range(0, len(rows), batch):
        rr = rows[i:i + batch]
        dev = next(model.parameters()).device
        f = W.features(proc, [amap[r["id"]] for r in rr], dev)
        seqs = [tok.prompt + r["tokens"] + [tok.eot] for r in rr]
        L = max(len(s) for s in seqs)
        ids = torch.tensor([s + [tok.eot] * (L - len(s)) for s in seqs], device=dev)
        la = model(input_features=f, decoder_input_ids=ids).logits.float().log_softmax(-1)
        lb = base_model(input_features=f, decoder_input_ids=ids).logits.float().log_softmax(-1)
        for j, s in enumerate(seqs):
            a_, b_ = la[j, 3:len(s) - 1], lb[j, 3:len(s) - 1]       # positions predicting content tokens + EOT
            vals.append((b_.exp() * (b_ - a_)).sum(-1).mean().item())
    return round(float(np.mean(vals)), 6)


# ======================================================================== commands
def cmd_base(c, a):
    """Base greedy decode on the 300 practice clips, keeping the last decoder layer's MLP input (for the gate) and
    the base transcripts (LoRA targets for speech)."""
    proc, model = c.pm
    W.gpu_setup()
    model.to(W.DEV)
    tok = W.Tok(model, proc)
    t0 = time.time()
    rows = {}
    for s in W.PRACTICE_SUITES:
        man, aud = W.load_suite(s)
        out = W.transcribe(model, proc, tok, list(aud), batch=a.batch, capture=True)
        rows[s] = [dict(id=m["id"], tokens=r["tokens"], text=r["text"], x=r["x"],
                        top_first=r["top_first"], nsp=r["nsp"]) for m, r in zip(man, out)]
        h = np.mean([bool(W.norm(r["text"])) for r in out])
        print(f"{s}: {len(out)} clips; non-empty {100 * h:.1f}%", flush=True)
    secs = time.time() - t0
    torch.save(dict(rows=rows, capture_seconds=round(secs, 1), model=c.model_name), c.runs / "practice_base.pt")
    print(f"practice capture {secs:.0f}s -> {c.runs / 'practice_base.pt'}")


def cmd_vad(c, a):
    p = WP / "runs" / "whisper_silence" / "vad_segments.json"      # model-independent; shared by --model runs
    cache = json.loads(p.read_text()) if p.exists() else {}
    t0 = time.time()
    for s in W.HELDOUT_SUITES:
        man, aud = W.load_suite(s)
        todo = [(m, x) for m, x in zip(man, aud) if m["id"] not in cache]
        for m, x in todo:
            cache[m["id"]] = W.vad_segments(x)
        print(f"{s}: VAD on {len(todo)} clips ({time.time() - t0:.0f}s)", flush=True)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(cache))
    ns = W.load_suite("nonspeech")[0]
    print("non-speech clips with no detected speech:", sum(not cache[m["id"]] for m in ns), "/", len(ns))


def fit_gate(c, items, cache, seed):
    """The recipe (fixed rules from the original run; nothing tuned on held-out data):
      x   = last decoder layer's final_layer_norm output (MLP input), base greedy decoding of the practice clips
      d   = unit(mean x[pos 3] over non-speech - mean x[pos 3] over speech/padded/bg)   (pos 3 = <|notimestamps|>,
            the query that picks the first content token)
      thr = midpoint of max speech projection (every word-predicting position) and min non-speech projection at pos 3
      j   = least-used unit of that MLP: argmin mean|gelu(fc1_j x + b_j)| * ||fc2[:, j]|| over all practice positions
      edit: fc1[j] = d, b1[j] = -thr, fc2[:, j] = 4 * unit((E[" ."] - mean E[competitors]) / final-LN gain),
            competitors = union of the top-5 first tokens on practice non-speech (after HF suppression), minus " ."
    The matched random edit rewrites the same unit with random directions of the same norms (same bias)."""
    t0 = time.time()
    proc, model = c.pm
    model.cpu()
    tok = W.Tok(model, proc)
    ids = {x["id"] for x in items}
    NS = [r for r in cache["rows"]["practice_nonspeech"] if r["id"] in ids]
    SP = [r for s in W.PRACTICE_SUITES[1:] for r in cache["rows"][s] if r["id"] in ids]
    LY = model.config.decoder_layers - 1
    P = f"model.decoder.layers.{LY}."
    Wt = {k: v.detach() for k, v in model.named_parameters()}
    x_ns = torch.stack([r["x"][3] for r in NS])
    x_sp = torch.stack([r["x"][3] for r in SP])
    diff = x_ns.mean(0) - x_sp.mean(0)
    d = diff / diff.norm()
    p_ns = x_ns @ d
    p_sp = torch.cat([r["x"][3:3 + len(r["tokens"])] @ d for r in SP])
    thr = (float(p_sp.max()) + float(p_ns.min())) / 2
    w1, b1, w2 = (Wt[P + k].clone() for k in ("fc1.weight", "fc1.bias", "fc2.weight"))
    use = torch.zeros(w1.shape[0])
    npos = 0
    for r in NS + SP:
        xx = r["x"][:4 + len(r["tokens"])]
        use += F.gelu(xx @ w1.T + b1).abs().sum(0)
        npos += len(xx)
    j = int((use / npos * w2.norm(dim=0)).argmin())
    dot = tok.encode(" .")[0]
    comp = sorted({t for r in NS for t in r["top_first"][:TOPK]} - {dot} - set(range(tok.eot, 10 ** 6)))
    E = Wt["model.decoder.embed_tokens.weight"]
    v = (E[dot] - E[comp].mean(0)) / Wt["model.decoder.layer_norm.weight"]
    g1, gb, g2 = w1.clone(), b1.clone(), w2.clone()
    g1[j], gb[j], g2[:, j] = d, -thr, GAIN * v / v.norm()
    gate = {P + "fc1.weight": g1, P + "fc1.bias": gb, P + "fc2.weight": g2}
    gen = torch.Generator().manual_seed(seed)
    r1, r2 = torch.randn(d.shape, generator=gen), torch.randn(v.shape, generator=gen)
    q1, qb, q2 = w1.clone(), b1.clone(), w2.clone()
    q1[j], qb[j], q2[:, j] = r1 / r1.norm() * d.norm(), -thr, GAIN * r2 / r2.norm()
    rnd = {P + "fc1.weight": q1, P + "fc1.bias": qb, P + "fc2.weight": q2}
    share = cache["capture_seconds"] * len(items) / sum(len(v_) for v_ in cache["rows"].values())
    meta = dict(layer=LY, unit=j, threshold=round(thr, 3), gain=GAIN, competitors=[tok.tokenizer.decode([t]) for t in comp],
                target=" .", entries_changed=W.count_changed(gate, model),
                ns_pos3_proj=[round(float(p_ns.min()), 2), round(float(p_ns.median()), 2), round(float(p_ns.max()), 2)],
                speech_proj_max=round(float(p_sp.max()), 2), practice_ns_below_thr=int((p_ns <= thr).sum()),
                n_items=len(items), n_nonspeech=len(NS), n_speech=len(SP),
                fit_seconds=round(time.time() - t0 + share, 1),
                fit_seconds_note="CPU fit time + this subset's share of the base practice decode (activation capture)")
    return gate, rnd, meta


def cmd_fit(c, a):
    items, cache = c.practice_items(a.n, a.seed)
    gate, rnd, meta = fit_gate(c, items, cache, seed=a.seed + 1)
    tg = tagname(a)
    W.save_edit(gate, c.runs / "edits" / f"gate{'_' + tg if tg else ''}.safetensors", meta)
    W.save_edit(rnd, c.runs / "edits" / f"random{'_' + tg if tg else ''}.safetensors",
                dict(meta, entries_changed=W.count_changed(rnd, c.pm[1]), kind="random control"))
    print("gate:", json.dumps(meta, ensure_ascii=False), flush=True)
    return meta


def train_lora(c, items, cache, out_dir, seed):
    """LoRA SFT of the decoder (encoder frozen) on the same practice clips the gate uses: non-speech -> " ." + EOT
    (the shortest output HF decoding allows; normalizes to empty), speech/padded/bg -> the base model's own greedy
    transcript + EOT. Loss on content tokens and EOT. Fixed recipe (patchkit's): r=16, alpha=32, dropout 0.05,
    lr 2e-4 AdamW, 3 epochs (min 30 steps), batch 4, 5 warm-up steps then linear decay, bf16 autocast."""
    from peft import LoraConfig, get_peft_model
    t0 = time.time()
    proc, model = W.load(c.model_name)
    W.gpu_setup()
    model.to(W.DEV)
    tok = W.Tok(model, proc)
    torch.manual_seed(seed)
    rng = random.Random(seed)
    dot = tok.encode(" .")[0]
    audio = {}
    for s in W.PRACTICE_SUITES:
        man, aud = W.load_suite(s)
        audio.update({m["id"]: x for m, x in zip(man, aud)})
    toks = {r["id"]: r["tokens"] for s in W.PRACTICE_SUITES for r in cache["rows"][s]}
    data = []
    for x in items:
        tgt = [dot] if x["suite"] == "practice_nonspeech" else toks[x["id"]]
        data.append((audio[x["id"]], tok.prompt + tgt + [tok.eot]))
    cfg = LoraConfig(r=LORA["r"], lora_alpha=LORA["alpha"], lora_dropout=LORA["dropout"], target_modules=LORA_TARGETS)
    pm = get_peft_model(model, cfg)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LORA["lr"], weight_decay=0.0)
    B = LORA["batch"]
    steps = max(LORA["min_steps"], math.ceil(LORA["epochs"] * len(data) / B))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 5) * max(0.0, 1 - s / steps))
    pm.train()
    losses, order = [], []
    for s in range(steps):
        if len(order) < B:
            order += rng.sample(range(len(data)), len(data))
        bi, order = order[:B], order[B:]
        with torch.no_grad():
            enc = W.encode(model, W.features(proc, [data[i][0] for i in bi]))
        L = max(len(data[i][1]) for i in bi)
        inp = torch.full((len(bi), L), tok.eot)
        lab = torch.full((len(bi), L), -100)
        for k, i in enumerate(bi):
            ids = data[i][1]
            inp[k, :len(ids)] = torch.tensor(ids)
            lab[k, 3:len(ids) - 1] = torch.tensor(ids[4:])        # position t predicts token t+1; from pos 3 on
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(encoder_outputs=(enc,), decoder_input_ids=inp.to(W.DEV), use_cache=False).logits
        loss = F.cross_entropy(logits.float().flatten(0, 1), lab.to(W.DEV).flatten(), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        losses.append(loss.item())
    pm.eval()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pm.save_pretrained(str(out_dir))
    share = cache["capture_seconds"] * len(items) / sum(len(v) for v in cache["rows"].values())
    meta = dict(LORA, target_modules=LORA_TARGETS, n_examples=len(data), steps=steps,
                trainable_params=sum(p.numel() for p in params), entries_changed=None,
                n_nonspeech=sum(x["suite"] == "practice_nonspeech" for x in items),
                loss_first=round(float(np.mean(losses[:5])), 4), loss_last=round(float(np.mean(losses[-5:])), 4),
                fit_seconds=round(time.time() - t0 + share, 1),
                fit_seconds_note="training time (incl. model load) + this subset's share of the base practice decode",
                n_items=len(items), seed=seed)
    (Path(out_dir) / "meta.json").write_text(json.dumps(meta, indent=1))
    merged = pm.merge_and_unload()
    merged.eval()
    print("lora:", json.dumps({k: meta[k] for k in ("n_examples", "steps", "trainable_params", "loss_first",
                                                    "loss_last", "fit_seconds")}), flush=True)
    return merged, meta


def cmd_lora(c, a):
    items, cache = c.practice_items(a.n, a.seed)
    tg = tagname(a)
    m, meta = train_lora(c, items, cache, c.runs / "lora" / (tg or "full"), a.seed)
    del m
    torch.cuda.empty_cache()


def arm_model(c, name, tag):
    """-> (model on DEV, meta, modifies_weights)."""
    sfx = f"_{tag}" if tag else ""
    proc, model = c.pm
    if name in ("gate", "random"):
        p = c.runs / "edits" / f"{name}{sfx}.safetensors"
        m = c.fresh_model()
        W.apply_edit(m, W.load_edit(p))
        return m.to(W.DEV), json.loads(p.with_suffix(".json").read_text()), True
    if name == "lora":
        from peft import PeftModel
        p = c.runs / "lora" / (tag or "full")
        m = PeftModel.from_pretrained(c.fresh_model(), str(p)).merge_and_unload().eval()
        return m.to(W.DEV), json.loads((p / "meta.json").read_text()), True
    return model.to(W.DEV), {}, False


def eval_on(c, model, arm, suites_data, base_rows, vad=None, thresholds=None, from_base=False):
    proc, base_model = c.pm
    tok = W.Tok(base_model, proc)
    rows = {}
    t0 = time.time()
    for s, (man, aud) in suites_data.items():
        if from_base:
            ids = {m["id"] for m in man}
            rr = [dict(r) for r in base_rows[s] if r["id"] in ids]
        else:
            rr = run_rows(model, proc, tok, s, man, list(aud), c.a.batch, vad=vad)
        if thresholds is not None:
            for r in rr:
                keep = "nsp" not in r or W.threshold_keep(r, thresholds)
                r["text"] = r["raw_text"] if keep else ""
                r["threshold_dropped"] = not keep
        rows[s] = rr
        print(f"  [{arm}] {s}: {len(rr)} clips, {time.time() - t0:.0f}s", flush=True)
    return rows, round(time.time() - t0, 1)


def cmd_eval(c, a):
    proc, base_model = c.pm
    W.gpu_setup()
    base_model.to(W.DEV)
    sd = load_suites(W.HELDOUT_SUITES)
    for name in a.arms.split(","):
        tag = a.tag
        arm = name + (f"_{tag}" if tag else "")
        vad = thr = None
        from_base = name.startswith("thresholds")
        if name == "thresholds":
            thr = W.OPENAI_DEFAULT
        elif name == "thresholds_tuned":
            thr = W.PRACTICE_TUNED
        elif name == "vad":
            vad = json.loads((WP / "runs" / "whisper_silence" / "vad_segments.json").read_text())
        base_rows = c.base_rows() if name != "base" else None
        model, meta, mod = arm_model(c, name if not from_base else "base", tag)
        if thr is not None:
            meta = dict(thresholds=thr)
        if vad is not None:
            meta = dict(vad=W.VAD_OPTS)
        rows, secs = eval_on(c, model, arm, sd, base_rows, vad=vad, thresholds=thr, from_base=from_base)
        if name == "base":
            for s, rr in rows.items():
                write_jsonl(c.runs / "eval" / "base" / f"{s}.jsonl", rr)
            base_rows = rows
        else:
            for s, rr in rows.items():
                write_jsonl(c.runs / "eval" / arm / f"{s}.jsonl", rr)
        metrics = score(rows, base_rows)
        kl = speech_kl(model, base_model, proc, W.Tok(base_model, proc), base_rows) if mod else \
            (None if vad is not None else 0.0)
        res = result(c, arm, metrics, kl, meta, n=sum(len(v) for v in rows.values()), eval_s=secs)
        (c.res / f"{arm}.json").write_text(json.dumps(res, indent=1, ensure_ascii=False))
        write_jsonl(c.res / "outputs" / f"{arm}.jsonl",
                    [dict(id=r["id"], suite=s, text=r["text"][:1500]) for s, rr in rows.items() for r in rr])
        print(arm, json.dumps(res["headline"]), "KL", kl, json.dumps(metrics), flush=True)
        if mod:
            del model
            torch.cuda.empty_cache()


def cmd_sweep(c, a):
    proc, base_model = c.pm
    W.gpu_setup()
    base_model.to(W.DEV)
    base_rows = c.base_rows()
    sd = load_suites(list(SWEEP_SUBSET), SWEEP_SUBSET)
    out_dir = c.res / "sweep"
    out_dir.mkdir(exist_ok=True)
    n_all = sum(len(v) for v in c.practice_cache()["rows"].values())
    for nn in a.ns.split(","):
        n = None if nn == "all" else int(nn)
        for seed in [int(s) for s in a.seeds.split(",")]:
            for meth in a.methods.split(","):
                if n is None and seed > 0 and meth == "gate":
                    continue                      # gate on all data is deterministic
                f = out_dir / f"{meth}_n{n or 'all'}_s{seed}.json"
                if f.exists():
                    continue
                t0 = time.time()
                items, cache = c.practice_items(n, seed)
                tg = f"sw_n{n or 'all'}_s{seed}"
                if meth == "gate":
                    gate, _, meta = fit_gate(c, items, cache, seed + 1)
                    model = c.fresh_model()
                    W.apply_edit(model, gate)
                    model.to(W.DEV)
                    base_model.to(W.DEV)
                else:
                    model, meta = train_lora(c, items, cache, c.runs / "lora" / tg, seed)
                rows, secs = eval_on(c, model, f"{meth}_{tg}", sd, base_rows)
                metrics = score(rows, base_rows)
                kl = speech_kl(model, base_model, proc, W.Tok(base_model, proc), base_rows)
                res = result(c, f"{meth}_{tg}", metrics, kl, meta, split="heldout_sweep_subset",
                             n=sum(len(v) for v in rows.values()), eval_s=secs)
                res.update(n_practice=n or n_all, seed=seed, method=meth,
                           subset_note="nonspeech 300 + padded[::2] 150 + clean[:150] + bg[::2] 150")
                f.write_text(json.dumps(res, indent=1, ensure_ascii=False))
                print("SWEEP", meth, n, seed, json.dumps(res["headline"]), "KL", kl,
                      f"{time.time() - t0:.0f}s", flush=True)
                del model
                torch.cuda.empty_cache()


def cmd_audio(c, a):
    """Export <=8 s 16 kHz mono WAV excerpts of held-out clips to site/audio/<id>.wav (for the blog)."""
    import soundfile as sf
    want = a.ids.split(",")
    out = WP / "site" / "audio"
    out.mkdir(parents=True, exist_ok=True)
    for s in W.HELDOUT_SUITES:
        man, aud = W.load_suite(s)
        for m, x in zip(man, aud):
            if m["id"] in want:
                x = np.asarray(x[:int(a.seconds * W.SR)], dtype=np.float32)
                sf.write(out / f"{m['id']}.wav", x, W.SR, subtype="PCM_16")
                print(m["id"], s, json.dumps({k: v for k, v in m.items() if k not in ("offset", "n")}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["base", "vad", "fit", "lora", "eval", "sweep", "audio"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--arms", default="base,thresholds,thresholds_tuned,vad,gate,random,lora")
    ap.add_argument("--ns", default="8,16,32,64,all")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--methods", default="gate,lora")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--ids", default="")
    ap.add_argument("--seconds", type=float, default=8.0)
    a = ap.parse_args()
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    c = Ctx(a)
    globals()["cmd_" + a.cmd](c, a)


if __name__ == "__main__":
    main()
