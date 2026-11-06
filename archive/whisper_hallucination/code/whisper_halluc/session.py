"""Agent sessions for the Whisper hallucination experiment.

  prepare: python whisper_halluc/session.py prepare --session runs/whisper/session_1
           -> PROMPT.md (from whisper_halluc/prompt_template.md) and an empty work/
  apply:   python whisper_halluc/session.py apply --session ...
           -> <session>/submitted_edit.safetensors (copy of work/edited/edit.safetensors; empty if the agent submitted
              nothing), with a summary of what changed
  random:  python whisper_halluc/session.py random --session ... --seed 0 --mode same|count
           -> <session>/random{seed}_{mode}_edit.safetensors: same tensors and the same L2 norm of the change per
              tensor, random Gaussian directions. mode same: exactly the agent's changed entries; mode count: the same
              number of entries per tensor at random positions.
  random --edit runs/whisper/head_scan/edit.safetensors --out runs/whisper/head_scan/random0_same_edit.safetensors
           -> a matched random control for any other edit file (e.g. the head scan).

All edits are {param_name: full new tensor} safetensors files; common.apply_edit is the shared apply helper used by
eval.py, agent_tools.py and ct2_check.py.
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


def base_params(model_name):
    global _base
    if _base is None:
        m = C.load_model(model_name)
        _base = {k: v.detach().float().clone() for k, v in C.param_dict(m).items()}
    return _base


def prepare(a):
    sess = Path(a.session)
    (sess / "work").mkdir(parents=True, exist_ok=True)
    text = (HERE / "prompt_template.md").read_text(encoding="utf-8")
    text = text.replace("{SESSION}", str(sess.resolve()).replace("\\", "/"))
    import os
    n_dec = sum(v.numel() for k, v in base_params(a.model).items() if k.startswith("model.decoder."))
    max_changed = int(os.environ.get("AGENT_MAX_CHANGED", int(0.01 * n_dec)))
    text = text.replace("{MAX_CHANGED}", f"{max_changed:,}").replace("{MODEL}", a.model)
    text = text.replace("{PASSES}", os.environ.get("AGENT_PASSES", "30"))
    (sess / "PROMPT.md").write_text(text, encoding="utf-8")
    print(f"session ready at {sess}")


def describe(path, model_name):
    e = C.read_edit(path)
    st = C.edit_stats(e, base_params(model_name))
    print(f"{path}: {sum(n for n, _ in st.values()):,} entries in {len(st)} tensors")
    for k, (n, l2) in st.items():
        print(f"  {k}: {n:,} entries, L2 {l2:.4f}")


def apply(a):
    sess = Path(a.session)
    src = sess / "work" / "edited" / "edit.safetensors"
    dst = sess / "submitted_edit.safetensors"
    if src.exists():
        shutil.copy(src, dst)
    else:
        C.write_edit(dst, {})
    describe(dst, a.model)


def random_edit(a):
    src = Path(a.edit) if a.edit else Path(a.session) / "submitted_edit.safetensors"
    edit = C.read_edit(src)
    base = base_params(a.model)
    g = torch.Generator().manual_seed(a.seed)
    out = {}
    for k, v in edit.items():
        key = "model.decoder.embed_tokens.weight" if k == "proj_out.weight" else k
        b = base[key]
        d = v.float() - b
        mask = (d != 0).flatten()
        n = int(mask.sum())
        if n == 0:
            continue
        idx = torch.randperm(b.numel(), generator=g)[:n] if a.mode == "count" else mask.nonzero().squeeze(1)
        r = torch.zeros(b.numel())
        r[idx] = torch.randn(n, generator=g)
        out[key] = b + r.view_as(b) * (d.norm() / r.norm())
    dst = Path(a.out) if a.out else Path(a.session) / f"random{a.seed}_{a.mode}_edit.safetensors"
    C.write_edit(dst, out)
    print(f"wrote {dst}")
    describe(dst, a.model)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "apply", "random"])
    ap.add_argument("--session", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["same", "count"], default="same")
    ap.add_argument("--edit", default=None, help="random: build the control for this edit file instead")
    ap.add_argument("--out", default=None, help="random: output path (default: in the session folder)")
    ap.add_argument("--model", default=C.DEFAULT_MODEL)
    a = ap.parse_args()
    if a.session is None and not (a.cmd == "random" and a.edit and a.out):
        ap.error("--session is required (or --edit and --out for random)")
    {"prepare": prepare, "apply": apply, "random": random_edit}[a.cmd](a)


if __name__ == "__main__":
    main()
