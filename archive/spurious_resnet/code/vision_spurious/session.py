"""Agent sessions for the spurious-repair experiment.

  prepare: python vision_spurious/session.py prepare --session runs/vision_spurious/session_1
           -> PROMPT.md (from vision_spurious/prompt_template.md) and an empty work/
  apply:   python vision_spurious/session.py apply --session ...
           -> <session>/submitted_edit.pt (copy of work/edited/edit.pt; {} if the agent submitted nothing)
  random:  python vision_spurious/session.py random --session ... --seed 0 --mode same|count
           -> <session>/random{seed}_{mode}_edit.pt: same tensors and same L2 norm of the change per tensor, random
              Gaussian directions. mode same: exactly the agent's changed entries; mode count: the same number of
              entries per tensor at random positions.
"""
import argparse
import shutil
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import common as C  # noqa: E402


def prepare(a):
    sess = Path(a.session)
    (sess / "work").mkdir(parents=True, exist_ok=True)
    text = (HERE / "prompt_template.md").read_text(encoding="utf-8")
    text = text.replace("{SESSION}", str(sess.resolve()).replace("\\", "/"))
    (sess / "PROMPT.md").write_text(text, encoding="utf-8")
    print(f"session ready at {sess}")


def apply(a):
    sess = Path(a.session)
    src = sess / "work" / "edited" / "edit.pt"
    if src.exists():
        shutil.copy(src, sess / "submitted_edit.pt")
    else:
        torch.save({}, sess / "submitted_edit.pt")
    e = torch.load(sess / "submitted_edit.pt", map_location="cpu")
    base = C.base_state()
    n = {k: int((v.float() != base[k].float()).sum()) for k, v in e.items()}
    print(f"submitted: {sum(n.values()):,} entries in {len(n)} tensors: {n}")


def random_edit(a):
    sess = Path(a.session)
    edit = torch.load(sess / "submitted_edit.pt", map_location="cpu")
    base = C.base_state()
    g = torch.Generator().manual_seed(a.seed)
    out = {}
    for k, v in edit.items():
        b = base[k].float()
        d = v.float() - b
        mask = (d != 0).flatten()
        n = int(mask.sum())
        if a.mode == "count":
            idx = torch.randperm(b.numel(), generator=g)[:n]
        else:
            idx = mask.nonzero().squeeze(1)
        r = torch.zeros(b.numel())
        r[idx] = torch.randn(n, generator=g)
        out[k] = b + r.view_as(b) * (d.norm() / r.norm())
    torch.save(out, sess / f"random{a.seed}_{a.mode}_edit.pt")
    print(f"wrote {sess / f'random{a.seed}_{a.mode}_edit.pt'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "apply", "random"])
    ap.add_argument("--session", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mode", choices=["same", "count"], default="same")
    a = ap.parse_args()
    {"prepare": prepare, "apply": apply, "random": random_edit}[a.cmd](a)


if __name__ == "__main__":
    main()
