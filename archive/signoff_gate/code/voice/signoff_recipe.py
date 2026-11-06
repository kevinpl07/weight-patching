"""Sign-off gate: the Whisper recipe (whisper_halluc/recipe_script.py) moved to an LLM's "Would you like me to..." endings.

Whisper: in the last decoder MLP, a difference-of-means "no speech" direction was read by one repurposed unit
(fc1 row = direction, bias = -threshold) whose fc2 column raised " .". Here:
  - positives: the position whose next token starts a sign-off (practice-phrase anchors only, voice/signoff.py)
  - negatives: positions whose next token starts a new *content* block; all other content positions are the
    false-firing check (the gate runs on every generated token, unlike Whisper's single query position)
  - read: the input of layer L's MLP (after pre_feedforward_layernorm)
  - Gemma's MLP has no biases: the threshold rides on a fitted direction m with m.x ~= 1 at every position,
    gate row = g*(d - theta*m), up row = k*m (so up ~= k), down column = a write vector that raises <end_of_turn>
    over the sign-off's usual first tokens, divided by the post-FFN norm gain so it survives that norm.

    python voice/signoff_recipe.py collect             # teacher-forced activations at the marked positions
    python voice/signoff_recipe.py fit [--layer 25]    # direction, separation, threshold, edit + random control
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import signoff_data as D  # noqa: E402
from signoff import signoff  # noqa: E402

WORK = C.RUNS / "signoff" / "recipe"
EDITS = C.RUNS / "signoff" / "edits"
LAYERS = (17, 21, 23, 24, 25)
import os  # noqa: E402
VERSION = os.environ.get("SIGNOFF_V", "2")
ALL_BLOCK_STARTS = VERSION in ("2", "3")  # v1 = first sign-off token only
ALL_PHRASES = VERSION == "3"  # v3: positives from every sign-off phrasing, not only the practice phrases
WORK = WORK.parent / {"1": "recipe", "2": "recipe_v2", "3": "recipe_v3"}[VERSION]


def positions(tok, prompt, text):
    """Concatenated ids plus marked positions (index of the hidden state that predicts the marked token)."""
    p_ids = tok(C.chat(tok, prompt), add_special_tokens=False).input_ids
    enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
    r_ids, offs = enc.input_ids, enc.offset_mapping
    # positives only where a *practice* phrase anchors the sign-off; but content always stops at *any* sign-off, so
    # held-out-phrase sign-offs are neither positives nor (mislabelled) negatives
    s_all, s_pr = signoff(text), signoff(text, ("practice",))
    s = dict(s_all, has=s_all["has"] and (s_pr["has"] or ALL_PHRASES))
    blocks = [m.end() for m in __import__("re").finditer(r"\n\s*\n", text)]  # char starts of blocks 2..k

    def first_tok_at(ch):  # first token that ends after char ch and is not pure whitespace
        for j, (a, b) in enumerate(offs):
            if b > ch and text[a:b].strip():
                return j
        return None

    P, N_para = [], []
    if s["has"]:
        j = first_tok_at(s["start"])
        if j is not None and j > 0:
            P.append(len(p_ids) + j - 1)
        if ALL_BLOCK_STARTS:  # v2: every non-list block start inside the sign-off (e.g. the anchor after "---")
            for ch in blocks:
                if ch > s["start"] and not __import__("re").match(r"\s*([-*•]|\d+[.)])\s", text[ch:ch + 8]):
                    jj = first_tok_at(ch)
                    if jj is not None and jj > 0 and len(p_ids) + jj - 1 not in P:
                        P.append(len(p_ids) + jj - 1)
    content_end_tok = first_tok_at(s_all["start"]) if s_all["has"] else len(r_ids)
    content_end_tok = len(r_ids) if content_end_tok is None else content_end_tok
    for ch in blocks:
        if s_all["has"] and ch >= s_all["start"]:
            break
        j = first_tok_at(ch)
        if j is not None and 0 < j < content_end_tok:
            N_para.append(len(p_ids) + j - 1)
    content = list(range(len(p_ids), len(p_ids) + max(0, content_end_tok - 1)))  # predicting content tokens
    eot = [len(p_ids) + len(r_ids) - 1] if not s_all["has"] else []  # predicts <end_of_turn> in clean answers
    return p_ids + r_ids, dict(pos=P, para=N_para, content=content, eot=eot), s


@torch.no_grad()
def collect(a):
    tok, model = C.load(C.MODEL)
    rows = C.read_jsonl(C.RUNS / "signoff" / "practice_base.jsonl")
    items = {it["id"]: it for it in D.load("practice")}
    store = {L: {k: [] for k in ("pos", "para", "content", "eot")} for L in LAYERS}
    acts = {L: {k: [] for k in ("pos", "para", "content")} for L in LAYERS}  # |MLP hidden| for least-used unit
    first_tokens = Counter()
    hooks, cur = [], {}
    for L in LAYERS:
        lyr = model.model.layers[L]
        hooks.append(lyr.pre_feedforward_layernorm.register_forward_hook(
            lambda mod, i, o, L=L: cur.__setitem__(("x", L), o[0].float().cpu())))
        hooks.append(lyr.mlp.down_proj.register_forward_hook(
            lambda mod, i, o, L=L: cur.__setitem__(("h", L), i[0][0].float().abs().cpu())))
    n_sig = 0
    with C.gpu("signoff recipe collect"):
        model.cuda()
        for r in rows:
            if r["truncated"]:
                continue
            ids, pos, s = positions(tok, items[r["id"]]["prompt"], r["text"])
            if s["has"] and pos["pos"]:
                n_sig += 1
                first_tokens[ids[pos["pos"][0] + 1]] += 1
            cur.clear()
            model(input_ids=torch.tensor([ids]).cuda(), use_cache=False)
            for L in LAYERS:
                x, h = cur[("x", L)], cur[("h", L)]
                for k, idx in pos.items():
                    if idx:
                        store[L][k].append(x[idx].half())
                for k in ("pos", "para", "content"):
                    if pos[k]:
                        acts[L][k].append(h[pos[k]].mean(0) if k == "content" else h[pos[k]].sum(0))
        model.cpu()
    for hk in hooks:
        hk.remove()
    WORK.mkdir(parents=True, exist_ok=True)
    out = {L: {k: torch.cat(v) if v else torch.empty(0, model.config.hidden_size) for k, v in store[L].items()}
           for L in LAYERS}
    use = {L: torch.stack(acts[L]["content"]).mean(0) for L in LAYERS}
    torch.save(dict(x=out, use=use, first_tokens=dict(first_tokens)), WORK / "acts.pt")
    print("answers with practice-phrase sign-off:", n_sig, "/", sum(not r["truncated"] for r in rows))
    print({k: len(v) for k, v in out[LAYERS[-1]].items()})
    print("sign-off first tokens:", [(tok.decode([t]), c) for t, c in first_tokens.most_common(12)])


def sep_stats(d, X):
    pr = {k: (v.float() @ d) for k, v in X.items() if len(v)}
    q = lambda t, p: torch.quantile(t, p).item()  # noqa: E731
    return {k: dict(n=len(v), min=v.min().item(), p05=q(v, .05), p50=q(v, .5), p95=q(v, .95), p999=q(v, .999),
                    max=v.max().item()) for k, v in pr.items()}


@torch.enable_grad()
def logreg_dir(X, steps=400, l2=1e-3, seed=0):
    """Unit direction from an L2 logistic probe: sign-off start (pos) vs every content position.
    Classes are re-weighted to equal mass. Returns (unit d, dict of fit info)."""
    P, N = X["pos"].float(), X["content"].float()
    mu, sd = N.mean(0), N.std(0) + 1e-3
    g = torch.Generator().manual_seed(seed)
    if len(N) > 60000:
        N = N[torch.randperm(len(N), generator=g)[:60000]]
    Z = torch.cat([(P - mu) / sd, (N - mu) / sd])
    y = torch.cat([torch.ones(len(P)), torch.zeros(len(N))])
    wts = torch.cat([torch.full((len(P),), 0.5 / len(P)), torch.full((len(N),), 0.5 / len(N))])
    w = torch.zeros(Z.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([w, b], max_iter=steps, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = (wts * torch.nn.functional.binary_cross_entropy_with_logits(Z @ w + b, y, reduction="none")).sum()
        loss = loss + l2 * w.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    d = (w.detach() / sd)  # back to raw activation space (the bias term is absorbed by the threshold)
    return d / d.norm(), dict(loss=closure().item())


def direction(X, kind):
    if kind == "dom":
        d = X["pos"].float().mean(0) - X["para"].float().mean(0)
        return d / d.norm()
    return logreg_dir(X)[0]


def fit(a):
    data = torch.load(WORK / "acts.pt")
    tok, model = C.load(C.MODEL)
    report = {}
    for L in LAYERS:
        X = data["x"][L]
        if a.cv:  # fit on 80% of positions, report separation on the other 20% (positions, not answers: rough)
            g = torch.Generator().manual_seed(0)
            tr, te = {}, {}
            for k, v in X.items():
                idx = torch.randperm(len(v), generator=g)
                cut = int(0.8 * len(v))
                tr[k], te[k] = v[idx[:cut]], v[idx[cut:]]
            d = direction(tr, a.dir)
            X = te
        else:
            d = direction(X, a.dir)
        st = sep_stats(d, X)
        gap = st["pos"]["p05"] - st["content"]["p999"]
        report[L] = dict(gap_p05_vs_content_p999=gap, **{k: {kk: round(vv, 2) for kk, vv in v.items()}
                                                         for k, v in st.items()})
        thr = st["content"]["p999"]
        rec = (X["pos"].float() @ d > thr).float().mean().item() if len(X["pos"]) else float("nan")
        thr2 = st["content"]["max"]
        rec2 = (X["pos"].float() @ d > thr2).float().mean().item() if len(X["pos"]) else float("nan")
        print(f"layer {L} [{a.dir}{' cv' if a.cv else ''}]: recall at content p99.9 {100 * rec:.0f}%, at content max "
              f"{100 * rec2:.0f}% | pos p50 {st['pos']['p50']:.2f} p05 {st['pos']['p05']:.2f}")
        print(f"   pos p05 {st['pos']['p05']:.2f} min {st['pos']['min']:.2f} | para p95 "
              f"{st['para']['p95']:.2f} max {st['para']['max']:.2f} | content p999 {st['content']['p999']:.2f} "
              f"max {st['content']['max']:.2f} | eot p50 {st.get('eot', {}).get('p50', float('nan')):.2f}")
    (WORK / "separation.json").write_text(json.dumps(report, indent=1))
    if a.layer is None:
        return
    L = a.layer
    X = data["x"][L]
    d = direction(X, a.dir)
    st = sep_stats(d, X)
    theta = a.theta if a.theta is not None else (st["pos"]["p05"] + st["content"]["max"]) / 2
    # constant direction: ridge least squares m with x.m ~= 1 over content positions
    A = X["content"].float()
    A = A[torch.randperm(len(A), generator=torch.Generator().manual_seed(0))[:20000]]
    lam = 1e-2 * (A.pow(2).sum() / len(A))
    m = torch.linalg.solve(A.T @ A + lam * torch.eye(A.shape[1]), A.T @ torch.ones(len(A)))
    allX = torch.cat([X[k].float() for k in X if len(X[k])])
    cm = allX @ m
    print(f"constant dir: x.m mean {cm.mean():.3f} std {cm.std():.3f} min {cm.min():.3f} max {cm.max():.3f}")
    lyr = model.model.layers[L]
    unit = int(data["use"][L].argmin())
    # write vector: raise <end_of_turn> over the usual sign-off first tokens, in residual space (tied embeddings)
    E = model.model.embed_tokens.weight.float()
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    ft = Counter({int(k): v for k, v in data["first_tokens"].items()})
    top = [t for t, _ in ft.most_common(10)]
    w = torch.tensor([ft[t] for t in top], dtype=torch.float)
    u = E[eot] - (E[top] * (w / w.sum())[:, None]).sum(0)
    u = u * (1 + model.model.norm.weight.float())  # final norm gain
    v = u / (1 + lyr.post_feedforward_layernorm.weight.float())  # undo the post-FFN norm gain
    v = v / v.norm()
    g, k, gain = a.g, a.k, a.gain
    gate = g * (d - theta * m)
    up = k * m
    down = gain * v
    edit = {f"model.layers.{L}.mlp.gate_proj.weight": lyr.mlp.gate_proj.weight.detach().clone(),
            f"model.layers.{L}.mlp.up_proj.weight": lyr.mlp.up_proj.weight.detach().clone(),
            f"model.layers.{L}.mlp.down_proj.weight": lyr.mlp.down_proj.weight.detach().clone()}
    edit[f"model.layers.{L}.mlp.gate_proj.weight"][unit] = gate.to(torch.bfloat16)
    edit[f"model.layers.{L}.mlp.up_proj.weight"][unit] = up.to(torch.bfloat16)
    edit[f"model.layers.{L}.mlp.down_proj.weight"][:, unit] = down.to(torch.bfloat16)
    EDITS.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    name = a.name or f"gate_L{L}"
    save_file({kk: vv.contiguous() for kk, vv in edit.items()}, str(EDITS / f"{name}.safetensors"))
    # matched random control: same unit, same three row/column norms, random directions
    gen = torch.Generator().manual_seed(1)
    rnd = {kk: vv.clone() for kk, vv in edit.items()}
    for kk, vec, ax in ((f"model.layers.{L}.mlp.gate_proj.weight", gate, 0),
                        (f"model.layers.{L}.mlp.up_proj.weight", up, 0),
                        (f"model.layers.{L}.mlp.down_proj.weight", down, 1)):
        rv = torch.randn(vec.shape, generator=gen)
        rv = rv / rv.norm() * vec.norm()
        if ax == 0:
            rnd[kk][unit] = rv.to(torch.bfloat16)
        else:
            rnd[kk][:, unit] = rv.to(torch.bfloat16)
    save_file({kk: vv.contiguous() for kk, vv in rnd.items()}, str(EDITS / f"{name}_random.safetensors"))
    meta = dict(layer=L, unit=unit, theta=theta, g=g, k=k, gain=gain, sep=st,
                first_tokens=[(tok.decode([t]), ft[t]) for t in top], changed=C.count_changed(edit, model))
    (EDITS / f"{name}.json").write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(json.dumps({kk: meta[kk] for kk in ("layer", "unit", "theta", "changed", "first_tokens")}, ensure_ascii=False))


@torch.no_grad()
def explain(a):
    """Top-scoring *content* positions for the layer-L probe direction, with text around them."""
    data = torch.load(WORK / "acts.pt")
    X = data["x"][a.layer]
    d = direction(X, a.dir)
    tok, model = C.load(C.MODEL)
    rows = [r for r in C.read_jsonl(C.RUNS / "signoff" / "practice_base.jsonl") if not r["truncated"]]
    items = {it["id"]: it for it in D.load("practice")}
    cur = {}
    hk = model.model.layers[a.layer].pre_feedforward_layernorm.register_forward_hook(
        lambda mod, i, o: cur.__setitem__("x", o[0].float().cpu()))
    hits = []
    with C.gpu("signoff recipe explain"):
        model.cuda()
        for r in rows:
            ids, pos, s = positions(tok, items[r["id"]]["prompt"], r["text"])
            model(input_ids=torch.tensor([ids]).cuda(), use_cache=False)
            sc = cur["x"] @ d
            for k in ("content", "pos"):
                for i in pos[k]:
                    hits.append((sc[i].item(), k, tok.decode(ids[max(0, i - 25):i + 1]), tok.decode(ids[i + 1:i + 9]),
                                 i - len(ids)))
        model.cpu()
    hk.remove()
    pos_scores = sorted(h[0] for h in hits if h[1] == "pos")
    print("pos score quantiles:", [round(pos_scores[int(q * (len(pos_scores) - 1))], 2) for q in (0, .1, .25, .5)])
    for h in sorted([h for h in hits if h[1] == "content"], reverse=True)[:a.n]:
        print(f"{h[0]:6.2f}  ...{h[2][-80:]!r}  -> {h[3]!r}   ({h[4]} from end)")


@torch.no_grad()
def probe(a):
    """Teacher-forced check of an edit on practice: P(<end_of_turn>) at sign-off positions, top-1 changes elsewhere."""
    tok, model = C.load(C.MODEL)
    rows = [r for r in C.read_jsonl(C.RUNS / "signoff" / "practice_base.jsonl") if not r["truncated"]][:a.n]
    items = {it["id"]: it for it in D.load("practice")}
    eot = tok.convert_tokens_to_ids("<end_of_turn>")
    edit = C.load_edit(a.edit)
    res = {"base": Counter(), "edit": Counter()}
    with C.gpu("signoff recipe probe"):
        for side in ("base", "edit"):
            orig = C.apply_edit(model, edit) if side == "edit" else None
            model.cuda()
            for r in rows:
                ids, pos, s = positions(tok, items[r["id"]]["prompt"], r["text"])
                lg = model(input_ids=torch.tensor([ids]).cuda(), use_cache=False).logits[0].float()
                top = lg.argmax(-1).cpu()
                pe = torch.softmax(lg, -1)[:, eot].cpu()
                c = res[side]
                for k in ("pos", "para", "content", "eot"):
                    for i in pos[k]:
                        c[f"{k}_n"] += 1
                        c[f"{k}_eot_top1"] += int(top[i] == eot)
                        c[f"{k}_peot"] += pe[i].item()
                res[side + "_top"] = res.get(side + "_top", []) + [top]
            model.cpu()
            if orig:
                C.apply_edit(model, orig)
    for side in ("base", "edit"):
        c = res[side]
        print(side, {k: f"eot-top1 {100 * c[k + '_eot_top1'] / max(1, c[k + '_n']):.1f}% "
                        f"mean P(eot) {c[k + '_peot'] / max(1, c[k + '_n']):.3f} (n={c[k + '_n']})"
                     for k in ("pos", "para", "content", "eot")})
    chg = sum(int((b != e).sum()) for b, e in zip(res["base_top"], res["edit_top"]))
    tot = sum(len(b) for b in res["base_top"])
    print(f"top-1 changed at {chg}/{tot} positions ({100 * chg / tot:.3f}%)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["collect", "fit", "probe", "explain"])
    ap.add_argument("--layer", type=int)
    ap.add_argument("--theta", type=float)
    ap.add_argument("--g", type=float, default=4.0)
    ap.add_argument("--k", type=float, default=4.0)
    ap.add_argument("--gain", type=float, default=200.0)
    ap.add_argument("--name")
    ap.add_argument("--edit")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--dir", choices=["dom", "logreg"], default="logreg")
    ap.add_argument("--cv", action="store_true")
    a = ap.parse_args()
    {"collect": collect, "fit": fit, "probe": probe, "explain": explain}[a.cmd](a)
