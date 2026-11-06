"""Agent sessions for the over-refusal experiment.

  prepare: python refusal/session.py prepare --model <hf id> --session runs/refusal/sessions/<model>/s1
           -> session.json (model, budgets), practice.json (copy of the labelled practice set), PROMPT.md (from
              refusal/prompt_template.md), empty work/
  apply:   python refusal/session.py apply --session ...
           -> <session>/submitted_edit.safetensors (copy of work/edited/edit.safetensors; empty if none) and
              <session>/arm_agent.json (eval config)
  random:  python refusal/session.py random --session ... --seed 0 --mode same|count
           -> <session>/random{seed}_{mode}_edit.safetensors + arm_random{seed}_{mode}.json: same tensors and the same
              L2 norm of the change per tensor as the agent's edit, Gaussian directions. mode same: exactly the
              agent's changed entries; mode count: the same number of entries per tensor at uniformly random positions.

Edit files are applied with common.apply_edit (also used by eval.py and agent_tools.py).
"""
import argparse
import shutil
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

PASSES = 30


def _base(model_name):
    m = C.load_model(model_name)
    return {k: v.detach() for k, v in C.params(m).items()}


def prepare(a):
    sess = Path(a.session).resolve()
    (sess / "work").mkdir(parents=True, exist_ok=True)
    src = C.DATA / C.slug(a.model) / "practice.json"
    if not src.exists():
        raise SystemExit(f"{src} missing: run `python refusal/data.py label --model {a.model}` first")
    shutil.copy(src, sess / "practice.json")
    m = C.load_model(a.model)
    n = C.n_params(m)
    max_changed = int(0.01 * n)
    cfg = m.config
    max_gen = 256 if n <= 2e9 else 96  # keeps one generate pass to ~1-2 GPU minutes on 3-4B models
    C.jdump(dict(model=a.model, max_passes=PASSES, max_changed=max_changed, n_params=n, max_gen=max_gen),
            sess / "session.json")
    from collections import Counter
    P = C.jload(sess / "practice.json")
    groups = Counter(p["group"] for p in P)
    sandwich = hasattr(C.layers(m)[0], "pre_feedforward_layernorm")
    repo = C.ROOT
    rel = Path(*[".."] * len(sess.relative_to(repo).parts))
    subs = {
        "{SESSION}": str(sess).replace("\\", "/"),
        "{REL}": str(rel).replace("\\", "/"),
        "{MODEL}": a.model,
        "{N_PARAMS}": f"{n:,}",
        "{MAX_CHANGED}": f"{max_changed:,}",
        "{PASSES}": str(PASSES),
        "{MAX_GEN}": str(max_gen),
        "{N_LAYERS}": str(len(C.layers(m))),
        "{HIDDEN}": str(cfg.hidden_size),
        "{N_OVER}": str(groups["over_refused"]), "{N_COMP}": str(groups["complied"]), "{N_HARM}": str(groups["harmful"]),
        "{TIE_NOTE}": ("**Tied embeddings:** the output layer (`lm_head`) is the same tensor as the input embedding "
                       "`model.embed_tokens.weight`. Edit it under that name only. Changing row t changes both what "
                       "token t writes into the network at the input and the logit of token t at the output."
                       if C.is_tied(m) else "The input embedding (`model.embed_tokens.weight`) and the output layer "
                       "(`lm_head.weight`) are separate tensors."),
        "{FUSED_NOTE}": ("Parameter names per layer: `self_attn.qkv_proj` (q, k and v fused, stacked in that order along "
                         "the output dimension), `self_attn.o_proj`, `mlp.gate_up_proj` (gate and up fused, gate first), "
                         "`mlp.down_proj`, `input_layernorm`, `post_attention_layernorm`."
                         if hasattr(C.layers(m)[0].self_attn, "qkv_proj") else
                         "Parameter names per layer: `self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj` and the "
                         "layer norms (see `T.info()`)."),
        "{NORM_NOTE}": ("Each attention and MLP block also has an RMSNorm *after* it (post_attention_layernorm, "
                        "post_feedforward_layernorm) before its output is added to the residual stream; RMSNorm "
                        "weights act as (1 + weight). The input embedding is multiplied by sqrt(hidden_size)."
                        if sandwich else "Attention and MLP outputs are added straight into the residual stream; "
                        "RMSNorm weights multiply the normalized input."),
    }
    text = (HERE / "prompt_template.md").read_text(encoding="utf-8")
    for k, v in subs.items():
        text = text.replace(k, v)
    (sess / "PROMPT.md").write_text(text, encoding="utf-8")
    print(f"session ready at {sess} (model {a.model}, {n:,} params, edit budget {max_changed:,} entries, "
          f"{PASSES} passes, practice {dict(groups)})")


def _arm(sess, name, edit, model):
    C.jdump(dict(name=name, model=model, edit=str(edit).replace("\\", "/"), system=None), sess / f"arm_{name}.json")


def apply(a):
    sess = Path(a.session).resolve()
    cfg = C.jload(sess / "session.json")
    src = sess / "work" / "edited" / "edit.safetensors"
    dst = sess / "submitted_edit.safetensors"
    if src.exists():
        shutil.copy(src, dst)
    else:
        C.save_edit_file({}, dst)
    e = C.load_edit(dst)
    m = C.load_model(cfg["model"])
    n = C.check_edit(m, e)
    P = C.params(m)
    norms = {k: round((v.float() - P[k].data.float()).norm().item(), 4) for k, v in e.items()}
    tag = a.name or f"agent_{sess.name}"
    _arm(sess, tag, dst, cfg["model"])
    print(f"submitted: {sum(n.values()):,} entries (limit {cfg.get('max_changed', int(0.01 * C.n_params(m))):,}) in {len(n)} tensors: {n}\n"
          f"delta norms: {norms}\neval config: {sess / f'arm_{tag}.json'}")


def _sample_positions(numel, n, g):
    """n distinct uniformly random flat indices out of numel (without materializing a full permutation)."""
    if n == 0:
        return torch.zeros(0, dtype=torch.long)
    if numel <= 50_000_000:
        return torch.randperm(numel, generator=g)[:n]
    picked = torch.zeros(0, dtype=torch.long)
    while len(picked) < n:
        cand = torch.randint(0, numel, (int((n - len(picked)) * 1.2) + 16,), generator=g)
        picked = torch.unique(torch.cat([picked, cand]))
    return picked[torch.randperm(len(picked), generator=g)[:n]]


def _bump(newb, bb, mask, sgn):
    """Entries in mask that bf16 rounding left equal to the base move by one bf16 step in the noise's direction, so
    the random edit changes exactly the intended entries."""
    stuck = mask & (newb == bb)
    if stuck.any():
        bits = bb.view(torch.int16).clone()
        up = (sgn > 0) == (bb.float() >= 0)  # moving away from zero = +1 in the bit pattern
        bits[stuck] += torch.where(up[stuck], 1, -1).to(torch.int16)
        fixed = bits.view(torch.bfloat16)
        zero = stuck & (bb == 0)
        fixed[zero] = (sgn[zero] * 1e-38).to(torch.bfloat16)
        newb = torch.where(stuck, fixed, newb)
    return newb


def random_edit(a):
    sess = Path(a.session).resolve()
    cfg = C.jload(sess / "session.json")
    edit = C.load_edit(sess / "submitted_edit.safetensors")
    m = C.load_model(cfg["model"])
    base = {k: v.data for k, v in C.params(m).items()}
    g = torch.Generator().manual_seed(a.seed)
    out = {}
    for k, v in edit.items():
        b = base[k].float()
        d = v.float() - b
        n = int((d != 0).sum())
        if n == 0:
            continue
        idx = _sample_positions(b.numel(), n, g) if a.mode == "count" else (d != 0).flatten().nonzero().squeeze(1)
        r = torch.zeros(b.numel())
        r[idx] = torch.randn(n, generator=g)
        mask = torch.zeros(b.numel(), dtype=torch.bool)
        mask[idx] = True
        bb = base[k].flatten()
        target = d.norm().item()
        scale = target / r.norm().item()
        for bump in (True, False):
            # bf16 rounding swallows small changes to large weights: bump them one bf16 step, then rescale until the
            # norm matches. If the forced steps alone overshoot the norm by >10% (tiny edits on large-valued tensors,
            # e.g. Gemma norm gains), matching the norm wins over matching the count.
            sc = scale
            for _ in range(6):
                newb = (b.flatten() + r * sc).to(torch.bfloat16)
                if bump:
                    newb = _bump(newb, bb, mask, torch.sign(r))
                got = (newb.float() - b.flatten()).norm().item()
                if got == 0 or abs(got / target - 1) < 0.01:
                    break
                sc *= target / got
            if abs(got / target - 1) < 0.1:
                break
        out[k] = newb.view_as(b)
        n_got = int((newb != bb).sum())
        print(f"{k}: {n:,} entries (changed after bf16: {n_got:,}), |delta| target {target:.4f}, got {got:.4f}")
    path = sess / f"random{a.seed}_{a.mode}_edit.safetensors"
    C.save_edit_file(out, path)
    _arm(sess, f"random{a.seed}_{a.mode}_{sess.name}", path, cfg["model"])
    print(f"wrote {path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "apply", "random"])
    ap.add_argument("--session", required=True)
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["same", "count"], default="same")
    ap.add_argument("--name", help="arm name for apply (default agent_<session folder>)")
    a = ap.parse_args()
    {"prepare": prepare, "apply": apply, "random": random_edit}[a.cmd](a)


if __name__ == "__main__":
    main()
