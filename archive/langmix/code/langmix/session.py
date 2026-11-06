"""Agent sessions for the language-mixing experiment (modelled on vision_spurious/session.py).

  prepare: python langmix/session.py prepare --session runs/langmix/sessions/s1
           -> PROMPT.md (from langmix/prompt_template.md, with paths and budgets filled in) and an empty work/
  apply:   python langmix/session.py apply --session runs/langmix/sessions/s1
           -> <session>/submitted_edit.safetensors (copy of work/edited/edit.safetensors; empty if the agent saved
              nothing) and a printout of the changed entries per tensor
  random:  python langmix/session.py random --session runs/langmix/sessions/s1 --seed 0 --mode same|count
           -> <session>/random{seed}_{mode}_edit.safetensors: the same tensors, and per tensor the same L2 norm of
              the change, in random Gaussian directions. mode same: exactly the agent's changed entries; mode count:
              the same number of entries per tensor at uniformly random positions.

Evaluate the files with langmix/eval.py --edit <file>. Both eval.py and agent_tools.py apply edits through
common.apply_edit (the one shared helper), so a submitted edit scores exactly as the agent tested it.
"""
import argparse
import shutil
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402

_base = None


def base_state(model):
    global _base
    if _base is None:
        _base = C.load_model(model).state_dict()
    return _base


def prepare(a):
    sess = Path(a.session)
    (sess / "work").mkdir(parents=True, exist_ok=True)
    m = C.load_model(a.model)
    n = sum(p.numel() for p in m.parameters())
    text = (HERE / "prompt_template.md").read_text(encoding="utf-8")
    rel = Path(*([".."] * len(sess.resolve().relative_to(C.ROOT).parts)))
    c = m.config
    tied = C.is_tied(m)
    desc = (f"**{a.model.split('/')[-1]}** in bf16, {n / 1e9:.2f}B parameters: a {type(m).__name__.replace('ForCausalLM', '')} "
            f"decoder, {c.num_hidden_layers} layers, hidden size {c.hidden_size:,}, {c.num_attention_heads} query heads and "
            f"{c.num_key_value_heads} key/value heads, SwiGLU MLPs of width {c.intermediate_size:,}, RMSNorm, a final norm "
            f"(`model.norm.weight`) and an output matrix `lm_head.weight` [{c.vocab_size:,} × {c.hidden_size:,}] that is "
            + ("**tied** to the input embeddings (one tensor: editing it changes both)." if tied
               else "**not** tied to the input embeddings."))
    scope = ("Any tensor may be edited, including `lm_head.weight` (tied to the input embeddings) and "
             "`model.norm.weight`." if tied else "Any tensor except the input embeddings (`model.embed_tokens.weight`) "
             "may be edited, including `lm_head.weight` and `model.norm.weight`.")
    text = text.replace("{MODEL_DESC}", desc).replace("{SCOPE}", scope)
    text = (text.replace("{SESSION}", str(sess.resolve()).replace("\\", "/"))
            .replace("{REPO_REL}", str(rel).replace("\\", "/"))
            .replace("{EDIT_BUDGET}", f"{int(0.005 * n):,}").replace("{PARAMS}", f"{n / 1e9:.2f}B"))
    (sess / "PROMPT.md").write_text(text, encoding="utf-8")
    print(f"session ready at {sess} (edit budget {int(0.005 * n):,} entries of {n:,})")


def apply(a):
    from safetensors.torch import save_file
    sess = Path(a.session)
    src = sess / "work" / "edited" / "edit.safetensors"
    dst = sess / "submitted_edit.safetensors"
    if src.exists():
        shutil.copy(src, dst)
    else:
        save_file({}, str(dst))
    e = C.load_edit(dst)
    n = C.count_changed(e, base_state(a.model))
    print(f"submitted: {sum(n.values()):,} entries in {len(n)} tensors: {n}")


def _round_nudge(b, r, dtype):
    """b + r rounded to dtype. bf16 rounding swallows the smallest random changes; those entries are moved by one
    bf16 step in the sign of r, so the random edit changes exactly as many entries as the agent's."""
    new = (b + r).to(dtype)
    stuck = (new.float() == b) & (r != 0)
    if stuck.any():
        bs = b[stuck]
        ulp = torch.where(bs == 0, torch.full_like(bs, 2.0 ** -126), torch.exp2(torch.floor(torch.log2(bs.abs())) - 7))
        nf = new.float()
        nf[stuck] = bs + torch.sign(r[stuck]) * ulp
        new = nf.to(dtype)
    return new


def random_edit(a):
    from safetensors.torch import save_file
    sess = Path(a.session)
    edit = C.load_edit(sess / "submitted_edit.safetensors")
    base = base_state(a.model)
    g = torch.Generator().manual_seed(a.seed)
    out = {}
    for k, v in edit.items():
        b = base[k].float()
        d = v.float() - b
        mask = (d != 0).flatten()
        n = int(mask.sum())
        if n == 0:
            continue
        idx = torch.randperm(b.numel(), generator=g)[:n] if a.mode == "count" else mask.nonzero().squeeze(1)
        r = torch.zeros(b.numel())
        r[idx] = torch.randn(n, generator=g)
        r = r.view_as(b) * (d.norm() / r.norm())
        scale = 1.0
        for _ in range(6):  # a few rescalings so the L2 norm after rounding/nudging matches the agent's
            new = _round_nudge(b, r * scale, base[k].dtype)
            scale *= float(d.norm() / (new.float() - b).norm())
        out[k] = new.contiguous()
    path = sess / f"random{a.seed}_{a.mode}_edit.safetensors"
    save_file(out, str(path))
    got = C.count_changed(out, base)
    norms = {k: round(float((out[k].float() - base[k].float()).norm()), 4) for k in out}
    ref = {k: round(float((edit[k].float() - base[k].float()).norm()), 4) for k in out}
    print(f"wrote {path}: {sum(got.values()):,} entries; L2 of change per tensor {norms} (agent: {ref})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "apply", "random"])
    ap.add_argument("--session", required=True)
    ap.add_argument("--model", default=C.MODEL)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["same", "count"], default="same")
    a = ap.parse_args()
    {"prepare": prepare, "apply": apply, "random": random_edit}[a.cmd](a)


if __name__ == "__main__":
    main()
