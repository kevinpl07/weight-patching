"""Arm 7, "it's just a script": the agents' over-refusal recipe families, built in closed form from practice data only.

  python refusal/recipe_script.py [--model microsoft/Phi-3-mini-4k-instruct] [--no-check] [--no-agents]

Data: the 300 practice prompts (runs/refusal/data/<model>/practice.json: 100 over_refused O, 100 complied C,
100 harmful H). Activations: residual stream at the last prompt token (`<|assistant|>`), index l = entering layer l
(common.last_token_resid), and the MLP activation (down_proj input) at the same token. dOC(l) = mean O - mean C.
Fixed rules (nothing tuned on held-out data, no dose sweep):

  recipe_shift   (sessions 4, 5)  L = 17, r = unit dOC(18); j = layer-17 neuron with the lowest coefficient of variation
                 (std/|mean| over all 300 prompts) among |mean| > 0.5; down_proj[:, j] -= shift * r / mean_j with
                 shift = 0.75 * |dOC(18)| (= 0.75 x the O-C gap along r, session 5's dose). One column.
  recipe_orth    (session 2)  o = dOC(20), h = unit(mean H - mean C at 20), w = unit(o - (o.h) h), sparsified to its
                 top-700 coordinates and renormalised (the dense version would change 138M entries > the 38.2M budget);
                 W' = W - w w^T W for o_proj and down_proj of layers 16-19.
  recipe_neurons (session 6)  u_l = unit dOC(l+1); per neuron j of layers 10-27, its write onto u_l, c = act_j (W_down[:, j].u_l);
                 keep neurons with mean_O c - mean_C c > max(0, mean_H c - mean_C c), take the top 300 by mean_O c - mean_C c,
                 and remove the u_l component from their down_proj columns.

Outputs: runs/refusal/recipe/<arm>.safetensors and arm configs runs/refusal/recipe/arm_<arm>.json for refusal/eval.py.
The verification block after the recipe (not part of it) compares each recipe with the agents' submitted edits and
runs a 60-prompt practice slice (every 5th prompt per group, greedy, 64 tokens).
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402  (also puts the repo root on sys.path)

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="microsoft/Phi-3-mini-4k-instruct")
ap.add_argument("--L", type=int, default=17)
ap.add_argument("--frac", type=float, default=0.75)
ap.add_argument("--orth-ref", type=int, default=20)
ap.add_argument("--orth-layers", default="16,17,18,19")
ap.add_argument("--k", type=int, default=700)
ap.add_argument("--neurons", type=int, default=300)
ap.add_argument("--out", default=str(C.RUNS / "recipe"))
ap.add_argument("--no-check", action="store_true")
ap.add_argument("--no-agents", action="store_true", help="skip generating with the agents' edits in the check")
a = ap.parse_args()

# ---------------------------------------------------------------- logic (30 lines, prints included)
P = C.jload(C.DATA / C.slug(a.model) / "practice.json"); G = np.array([p["group"] for p in P])
tok, model = C.load(a.model); W = {k: v.detach() for k, v in C.params(model).items()}
LA, acts = range(10, 28), {l: [] for l in range(10, 28)}
hooks = [model.model.layers[l].mlp.down_proj.register_forward_pre_hook(lambda m, i, l=l: acts[l].append(i[0][:, -1].float().cpu())) for l in LA]
with C.on_gpu(model, "refusal recipe_script capture"):  # one hold, 300 short prompts (on_gpu caps memory at 0.85)
    X = C.last_token_resid(model, tok, [p["prompt"] for p in P])  # [300, 33, hidden]
[h.remove() for h in hooks]; A = {l: torch.cat(acts[l]) for l in LA}
mean = lambda x, g: x[torch.from_numpy(G == g)].mean(0)  # noqa: E731
dOC = lambda l: mean(X[:, l], "over_refused") - mean(X[:, l], "complied")  # noqa: E731
unit, down, o_proj = (lambda v: v / v.norm()), "model.layers.{}.mlp.down_proj.weight", "model.layers.{}.self_attn.o_proj.weight"
# (a) recipe_shift: a near-constant neuron as a bias carrier, pushing -shift along the refusal direction
r, shift, mu = unit(dOC(a.L + 1)), a.frac * float(dOC(a.L + 1).norm()), A[a.L].mean(0)
j = int(torch.where(mu.abs() > 0.5, A[a.L].std(0) / mu.abs(), torch.inf).argmin())
Wd = W[down.format(a.L)].float().clone(); Wd[:, j] -= shift * r / mu[j]; E = {"recipe_shift": {down.format(a.L): Wd}}
# (b) recipe_orth: project the over-refusal-only component w (o orthogonal to h) out of layers 16-19's writers
o, h = dOC(a.orth_ref), unit(mean(X[:, a.orth_ref], "harmful") - mean(X[:, a.orth_ref], "complied"))
w = unit(o - (o @ h) * h); top = w.abs().topk(a.k).indices; wk = torch.zeros_like(w); wk[top] = w[top]; wk = unit(wk)
E["recipe_orth"] = {n.format(l): W[n.format(l)].float() - torch.outer(wk, wk @ W[n.format(l)].float())
                    for l in map(int, a.orth_layers.split(",")) for n in (o_proj, down)}
# (c) recipe_neurons: drop the refusal-direction component of the top O-preferring refusal writers
U, cand, E["recipe_neurons"] = {l: unit(dOC(l + 1)) for l in LA}, [], {}
for l in LA:
    c = A[l] * (U[l] @ W[down.format(l)].float())  # [300, inter]: each neuron's write onto u_l
    dO, dH = mean(c, "over_refused") - mean(c, "complied"), mean(c, "harmful") - mean(c, "complied")
    cand += [(float(dO[i]), l, i) for i in torch.nonzero((dO > 0) & (dO > dH)).flatten().tolist()]
for _, l, i in sorted(cand, reverse=True)[:a.neurons]:
    Wn = E["recipe_neurons"].setdefault(down.format(l), W[down.format(l)].float().clone()); Wn[:, i] -= (Wn[:, i] @ U[l]) * U[l]
for name, e in E.items():
    C.save_edit_file(e, Path(a.out) / f"{name}.safetensors")
    C.jdump(dict(name=name, model=a.model, edit=(Path(a.out) / f"{name}.safetensors").as_posix(), system=None), Path(a.out) / f"arm_{name}.json")
    print(f"{name}: {len(e)} tensors, {sum(C.check_edit(model, e).values()):,} entries changed")
print(f"shift: L{a.L} neuron {j}, mean act {mu[j]:.3f}, cv {float(A[a.L][:, j].std() / mu[j].abs()):.3f}, |dOC(18)| {shift / a.frac:.2f}, shift {shift:.2f}; "
      f"orth: |w|/|o| {float((o - (o @ h) * h).norm() / o.norm()):.2f}, cos(w_700, w) {float(wk @ w):.3f}; neurons: {len(cand)} candidates")

# ---------------------------------------------------------------- verification (not part of the recipe)
if not a.no_check:
    import judge as J
    from safetensors import safe_open
    S = C.RUNS / "sessions" / C.slug(a.model)
    E = {k: {n: v.to(torch.bfloat16) for n, v in e.items()} for k, e in E.items()}  # what was saved; halves the RAM
    dl = lambda v, n: v.float() - W[n].float()  # noqa: E731  (W is still the base model here; the Swap below changes it)
    nr = {k: sum(float(dl(v, n).pow(2).sum()) for n, v in e.items()) ** 0.5 for k, e in E.items()}
    mine = {(l, i) for _, l, i in sorted(cand, reverse=True)[:a.neurons]}
    fam = {1: "recipe_orth", 2: "recipe_orth", 3: "recipe_shift", 4: "recipe_shift", 5: "recipe_shift", 6: "recipe_neurons"}
    print("\nagent edit vs recipe: tensors | entries changed | Frobenius cos of the deltas | mean |cos| of the agent's "
          "write direction (largest delta column) with r = unit dOC(18), with w_700, with unit dOC(own layer + 1)")
    for s, f in fam.items():
        dot = na = 0.0; ent = 0; cr, cw, cl, cols, extra = [], [], [], set(), []
        with safe_open(str(S / f"session_{s}" / "submitted_edit.safetensors"), "pt") as fh:
            for n in fh.keys():
                v, l = dl(fh.get_tensor(n), n), int(n.split(".")[2])
                ent += int((v != 0).sum()); na += float(v.pow(2).sum())
                dot += float((v * dl(E[f][n], n)).sum()) if n in E[f] else 0.0
                u = unit(v[:, v.norm(dim=0).argmax()])
                cr.append(abs(float(u @ r))); cw.append(abs(float(u @ wk))); cl.append(abs(float(u @ unit(dOC(l + 1)))))
                nz = torch.nonzero(v.abs().sum(0)).flatten().tolist(); cols |= {(l, jj) for jj in nz}
                if s in (4, 5):
                    for jj in nz:
                        x = A[l][:, jj] if l in A else torch.full((2,), float("nan"))
                        extra.append(f"L{l} n{jj}: mean act {float(x.mean()):+.3f}, cv {float(x.std() / x.mean().abs()):.3f}, "
                                     f"rank by cv at L{l} {int((A[l].std(0) / A[l].mean(0).abs() < x.std() / x.mean().abs())[A[l].mean(0).abs() > 0.5].sum()) if l in A else -1}, "
                                     f"cos(delta column, -r) {float(unit(v[:, jj]) @ -r):+.3f}, |delta column| {float(v[:, jj].norm()):.2f}")
                if s == 2 and not extra:
                    extra.append(f"cos(agent u, recipe w_700) {float(u @ wk):+.3f}, top-{a.k} coordinate overlap "
                                 f"{len(set(top.tolist()) & set(u.abs().topk(a.k).indices.tolist()))}/{a.k}")
        print(f"s{s} vs {f}: {len(cr)} | {ent:,} | {dot / (na ** 0.5 * nr[f]):+.3f} | r {np.mean(cr):.3f}  w {np.mean(cw):.3f}  "
              f"own {np.mean(cl):.3f} | layers {sorted({l for l, _ in cols})}")
        if s == 6:
            extra.append(f"neuron overlap: {len(cols & mine)} of the recipe's {len(mine)} (session 6 edited {len(cols)} columns)")
        for e in extra:
            print("    " + e)
    print(f"recipe delta norms: " + ", ".join(f"{k} {v:.2f}" for k, v in nr.items()))
    idx = [i for g in ("over_refused", "complied", "harmful") for i in np.nonzero(G == g)[0][::5].tolist()]
    prompts = [C.chat(tok, P[i]["prompt"]) for i in idx]
    cnt = lambda ref, g: sum(x for x, i in zip(ref, idx) if G[i] == g)  # noqa: E731
    show = lambda nm, ref: print(f"  {nm:16s} refused (any): O {cnt(ref, 'over_refused')}/20  C {cnt(ref, 'complied')}/20  "  # noqa: E731
                                 f"H {cnt(ref, 'harmful')}/20", flush=True)
    print("\npractice slice (every 5th practice prompt per group; greedy, 64 new tokens; base = the stored base labels)")
    show("base", [P[i]["base_label"] != "full_compliance" for i in idx])
    sw = C.Swap(model)
    runs = list(E.items()) + ([] if a.no_agents else [(f"agent_s{s}", S / f"session_{s}" / "submitted_edit.safetensors") for s in (5, 4, 2, 6)])
    with C.on_gpu(model, "refusal recipe_script check"):  # one hold for all arms (Swap edits the weights on the GPU)
        for nm, e in runs:
            sw.set_edit(e); sw.use("edited")
            res = C.generate(model, tok, prompts, max_new=64, batch=64, verbose=False)
            sw.use("base")
            show(nm, [J.classify(t) != "full_compliance" for t, *_ in res])
    print(f"GPU seconds inside the lock: {C.on_gpu.seconds:.0f}")
