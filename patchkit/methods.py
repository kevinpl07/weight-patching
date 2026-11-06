"""The fix methods compared on every task.

    base    the unmodified model
    prompt  the prompt-engineering fix: the task's system_fix added to the system prompt
    gate    the one-neuron gate: probe a direction the model already computes, repurpose the least-used neuron of one
            MLP layer to fire on it (above a threshold no ordinary position reaches) and push the logits toward the
            task's target opener. ~3 x d_model entries change. No gradient descent touches the model.
    random  the same neuron rewritten with random vectors of the same norms (is it just the perturbation?)
    lora    the training baseline: LoRA SFT (r=16, all attention+MLP projections, 3 epochs) on the SAME practice
            prompts, with targets the model writes itself (positives: continuation after the target opener;
            negatives: its own base reply). Fixed recipe, not tuned on held-out data.
"""
import contextlib
import json
import math
import random
import time
from collections import Counter
from pathlib import Path

import torch

from . import llm as L


# ======================================================================== gate
def layer_index(model, layer):
    """int: layer index (negative counts from the end); float in (0, 1]: fraction of depth (1.0 = last layer)."""
    n = len(L.text_model(model).layers)
    if isinstance(layer, float):
        return int(round(layer * (n - 1)))
    assert -n <= layer < n, f"layer {layer} out of range for {n} layers"
    return layer % n


def collect_sites(task, tok, items, base):
    sites = []
    for x in items:
        sites += [dict(s, id=x["id"]) for s in task.probe_sites(tok, x, base[x["id"]]["base"])]
    return sites


def capture_sites(model, tok, sites, layers, batch=8):
    X = L.capture(model, tok, [s["seq"] for s in sites], layers, positions=[s["pos"] for s in sites], batch=batch)
    lab = torch.tensor([s["label"] for s in sites for _ in s["pos"]])
    kind = [s["kind"] for s in sites for _ in s["pos"]]
    return X, lab, kind


def next_tokens(tok, sites):
    out = []
    for s in sites:
        ids = tok(s["seq"], add_special_tokens=False).input_ids
        out += [ids[p + 1] if p + 1 < len(ids) else None for p in s["pos"]]
    return out


def target_ids(tok, targets):
    out = []
    for t in targets:
        out.append(L.eot_id(tok) if t == "<eot>" else tok.encode(t, add_special_tokens=False)[0])
    return out


@torch.no_grad()
def signal(task, model, tok, items, base, layers):
    sites = collect_sites(task, tok, items, base)
    X, lab, kind = capture_sites(model, tok, sites, layers)
    dec = torch.tensor([k == "decision" for k in kind])
    rep = {}
    for Lr in layers:
        P, N = X[Lr][dec & (lab == 1)], X[Lr][dec & (lab == 0)]
        rep[Lr] = round(L.cv_auc(P, N), 4) if len(P) >= 5 and len(N) >= 5 else None
    return rep, dict(pos=int((dec & (lab == 1)).sum()), neg=int((dec & (lab == 0)).sum()),
                     content=int((~dec).sum()))


def fit_gate(task, model, tok, items, base, seed=1, calib=None):
    """-> (edit, random_edit, meta). items: the practice items to fit on.
    calib: optional label-free "ordinary text" sites (generic chat replies, see run.Ctx.calib). They count as content
    negatives (the threshold sits above every ordinary position) and anchor the constant direction m, so neither
    depends on how many labelled practice items there are."""
    t0 = time.time()
    Lr = layer_index(model, task.gate_layer)
    sites = collect_sites(task, tok, items, base)
    X, lab, kind = capture_sites(model, tok, sites, [Lr])
    X = X[Lr]
    if calib:
        Xc = capture_sites(model, tok, calib, [Lr])[0][Lr]
    else:
        Xc = X[:0]
    dec = torch.tensor([k == "decision" for k in kind])
    P = X[lab == 1]
    N_dec, N_con = X[dec & (lab == 0)], torch.cat([X[~dec & (lab == 0)], Xc])
    d = L.logreg_dir(P, torch.cat([N_dec, N_con]))
    sp, sn, sc = P @ d, N_dec @ d, N_con @ d
    hi = [torch.quantile(sn, .95).item()] if len(sn) else []
    hi += [sc.max().item()] if len(sc) else []
    theta = max(hi) + task.theta_margin
    m = L.const_dir(torch.cat([X, Xc]))
    # least-used neuron of the layer on the practice (+ calibration) sequences
    seqs = list({s["seq"] for s in sites})[:200] + [s["seq"] for s in (calib or [])]
    unit = int(L.unit_usage(model, tok, seqs, Lr, batch=8).argmin())
    # write vector: raise the target opener(s), lower what the model would otherwise say at the positive sites
    # only where the base reply actually shows the failure: a model that already handles some positives correctly
    # must not have its own correct openers suppressed
    failed = {x["id"] for x in items if task.positive(x) and task.failed(x, base[x["id"]]["base"])}
    site_ids = [s_["id"] for s_ in sites for _ in s_["pos"]]
    nxt = [t for t, l_, i in zip(next_tokens(tok, sites), lab.tolist(), site_ids)
           if l_ == 1 and t is not None and i in failed]
    tgt = target_ids(tok, task.gate_targets)
    cnt = Counter(t for t in nxt if t not in tgt)
    top = [t for t, _ in cnt.most_common(8)]
    v = L.write_vec(model, tok, Lr, tgt, top, [cnt[t] for t in top] if top else None)
    edit, rnd = L.build_edit(model, Lr, unit, d, theta, m, v, g=task.gate_g, k=task.gate_k, gain=task.gate_gain,
                             seed=seed)
    sd = dict(model.named_parameters())
    changed = L.count_changed(edit, {k: sd[k].detach().cpu() for k in edit})
    meta = dict(layer=Lr, unit=unit, theta=round(theta, 3), targets=[tok.decode([t]) for t in tgt],
                suppress=[tok.decode([t]) for t in top], entries_changed=changed,
                recall_practice=round((sp > theta).float().mean().item(), 3),
                neg_fire_practice=round((sn > theta).float().mean().item(), 3) if len(sn) else None,
                content_fire_practice=round((sc > theta).float().mean().item(), 5) if len(sc) else None,
                n_pos=len(P), n_neg=len(sn), n_content=len(sc), n_calib_seqs=len(calib or []), n_items=len(items), fit_seconds=round(time.time() - t0, 1))
    return edit, rnd, meta


# ======================================================================== LoRA
LORA_TARGETS = r".*(language_model|model)\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"


def train_lora(task, model, tok, items, base, out_dir, seed=0, r=16, alpha=32, lr=2e-4, epochs=3, min_steps=30,
               batch=4, max_len=768, replay=None, save_at=()):
    """replay: optional label-free (chat-formatted prompt, base reply) pairs mixed in to preserve ordinary behaviour
    (the same generic chat replies the gate uses for calibration)."""
    from peft import LoraConfig, get_peft_model
    t0 = time.time()
    torch.manual_seed(seed)
    rng = random.Random(seed)
    eot = tok.decode([L.eot_id(tok)])
    data = []
    for x in items:
        b = base[x["id"]]
        tgt = task.sft_target(x, b["base"], b.get("forced"))
        if tgt is None:
            continue
        p = task.render(tok, x)
        pi = tok(p, add_special_tokens=False).input_ids
        ri = tok(tgt + eot, add_special_tokens=False).input_ids
        ids = (pi + ri)[:max_len]
        data.append((ids, len(pi)))
    n_task = len(data)
    for p, rep in replay or []:
        pi = tok(p, add_special_tokens=False).input_ids
        ri = tok(rep + eot, add_special_tokens=False).input_ids
        data.append(((pi + ri)[:max_len], len(pi)))
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.05, target_modules=LORA_TARGETS, task_type="CAUSAL_LM")
    pm = get_peft_model(model, cfg)
    pm.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    params = [p for p in pm.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    steps = max(min_steps, math.ceil(epochs * len(data) / batch))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 5) * max(0.0, 1 - s / steps))
    losses = []
    order = []
    for s in range(steps):
        if len(order) < batch:
            order += rng.sample(range(len(data)), len(data))
        bi, order = order[:batch], order[batch:]
        mx = max(len(data[i][0]) for i in bi)
        inp = torch.full((len(bi), mx), tok.pad_token_id)
        lab = torch.full((len(bi), mx), -100)
        att = torch.zeros((len(bi), mx), dtype=torch.long)
        for j, i in enumerate(bi):
            ids, n_p = data[i]
            inp[j, :len(ids)] = torch.tensor(ids)
            lab[j, n_p:len(ids)] = torch.tensor(ids[n_p:])
            att[j, :len(ids)] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = pm(input_ids=inp.to(L.DEV), attention_mask=att.to(L.DEV), labels=lab.to(L.DEV), use_cache=False)
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        losses.append(out.loss.item())
        if s + 1 in save_at:
            pm.save_pretrained(Path(out_dir) / f"step{s + 1}")
            (Path(out_dir) / f"step{s + 1}" / "meta.json").write_text(json.dumps(dict(
                step=s + 1, trainable_params=sum(p.numel() for p in params), n_items=len(items))))
    pm.eval()
    if hasattr(model, "gradient_checkpointing_disable"):
        model.gradient_checkpointing_disable()
    for p in params:
        p.data = p.data.to(torch.bfloat16)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    pm.save_pretrained(out_dir)
    n_train = sum(p.numel() for p in params)
    meta = dict(n_examples=n_task, n_replay=len(data) - n_task, steps=steps, lr=lr, r=r, epochs=epochs, trainable_params=n_train,
                loss_first=round(sum(losses[:5]) / max(1, len(losses[:5])), 4),
                loss_last=round(sum(losses[-5:]) / max(1, len(losses[-5:])), 4),
                fit_seconds=round(time.time() - t0, 1), n_items=len(items))
    (Path(out_dir) / "meta.json").write_text(json.dumps(meta, indent=1))
    pm.unload()  # restore the plain base model (adapter saved to disk)
    return meta


# ======================================================================== arms
class Arm:
    """A way of running the model: base, prompt fix, weight edit, or LoRA adapter. on()/off() are exact inverses."""

    def __init__(self, name, fix=False, edit=None, lora=None, meta=None):
        self.name, self.fix, self.edit, self.lora, self.meta = name, fix, edit, lora, meta or {}
        self._orig, self._pm = None, None

    def on(self, model):
        if self.edit is not None:
            self._orig = L.apply_edit(model, self.edit)
        if self.lora is not None:
            from peft import PeftModel
            self._pm = PeftModel.from_pretrained(model, self.lora)
            self._pm.eval()

    def off(self, model):
        if self._orig is not None:
            L.apply_edit(model, self._orig)
            self._orig = None
        if self._pm is not None:
            self._pm.unload()
            self._pm = None

    @contextlib.contextmanager
    def as_base(self, model):
        """Temporarily the base model (for KL)."""
        if self._pm is not None:
            with self._pm.disable_adapter():
                yield
        elif self._orig is not None:
            L.apply_edit(model, self._orig)
            try:
                yield
            finally:
                L.apply_edit(model, self.edit)
        else:
            yield

    @property
    def modifies_weights(self):
        return self.edit is not None or self.lora is not None


@torch.no_grad()
def kl_arm(model, tok, arm, seqs, prompt_lens, batch=4):
    """Mean per-token KL(base || arm) on teacher-forced base chat replies."""
    tok.padding_side = "right"
    vals = []
    try:
        for i in range(0, len(seqs), batch):
            enc = tok(seqs[i:i + batch], return_tensors="pt", padding=True, add_special_tokens=False).to(L.DEV)
            with arm.as_base(model):
                a = torch.log_softmax(model(**enc).logits.float(), -1)
            b = torch.log_softmax(model(**enc).logits.float(), -1)
            for j, n in enumerate(enc.attention_mask.sum(1).tolist()):
                p0 = prompt_lens[i + j]
                x, y = a[j, p0 - 1:n - 1], b[j, p0 - 1:n - 1]
                vals.append((x.exp() * (x - y)).sum(-1).mean().item())
            del a, b
    finally:
        tok.padding_side = "left"
    return sum(vals) / len(vals)
