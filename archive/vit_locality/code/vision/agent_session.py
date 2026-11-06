"""Set up an agent session from a ViT checkpoint, apply its edit, and build the control checkpoints.

  prepare: python vision/agent_session.py prepare --run runs/vision_vit/base/s0 --step 1000 \
               --session runs/vision_vit/s0_step1000
           -> base.pt (weights only), session.json, loss_so_far.csv, work/
  Every checkpoint below is written to runs/vision_vit/arms/<session name>/ (ARMS), outside the agent's folder.
  apply:   python vision/agent_session.py apply --session <session>
           -> ARMS/edited_step_XXXXX.pt: the full original checkpoint (optimizer state included) with the
              agent's tensors written in; copies the edit to <session>/submitted_edit.pt
  random:  python vision/agent_session.py random --session <session> --seed 0
           -> ARMS/random{seed}_step_XXXXX.pt: same tensors, same number of changed entries per tensor
              and same L2 norm of the change per tensor, at random positions with random directions
  mimetic: python vision/agent_session.py mimetic --session <session>
           -> ARMS/mimetic_step_XXXXX.pt: scripted reference fix (vision/mimetic.py) on every layer
  none:    python vision/agent_session.py none --session <session>
           -> ARMS/none_step_XXXXX.pt: the unedited checkpoint (a copy, so every arm resumes the same way)
Resume any of them with: vision/train.py --init <ckpt> --out ARMS/arm_<name>  (or vision/run_arms.sh)
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
TRAIN_NPZ = HERE / "data_cifar" / "cifar10_train_only.npz"


def _train_only_npz():
    """The agent's data file: training split only (the test set must stay invisible to the agent)."""
    if not TRAIN_NPZ.exists():
        d = np.load(HERE / "data_cifar" / "cifar10_uint8.npz")
        np.savez(TRAIN_NPZ, x_train=d["x_train"], y_train=d["y_train"])
    return TRAIN_NPZ


def prepare(a):
    run, sess = Path(a.run), Path(a.session)
    ck_path = run / "ckpt" / f"step_{a.step:05d}.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    sess.mkdir(parents=True, exist_ok=True)
    (sess / "work").mkdir(exist_ok=True)
    torch.save({"model": ck["model"], "cfg": ck["cfg"], "step": ck["step"]}, sess / "base.pt")
    args = ck["args"]
    spe = 50000 // args["batch"]
    recipe = dict(
        optimizer=f"AdamW betas (0.9, 0.999), weight decay {args['wd']} on weight matrices and the patch-embedding "
                  "kernel (not on biases, LayerNorms or pos_embed), grad-norm clip 1.0, bf16 autocast",
        lr=f"peak {args['lr']}, linear warmup {args['warmup']} steps, cosine decay to "
           f"{args['min_lr_frac']} x peak at step {args['steps']}",
        batch=f"{args['batch']} images per step ({spe} steps per epoch, the last partial batch of each epoch dropped)",
        data="CIFAR-10 training set (50,000 images, 10 classes), per-channel mean/std normalization, random crop "
             "32x32 from the 4-pixel zero-padded image and random horizontal flip; no other augmentation, "
             "no label smoothing, no dropout",
        loss="cross-entropy on the logits",
        note="after your edit, training resumes from this step with the same optimizer state and the same "
             "sequence of batches (including the same augmentation draws), until total_steps")
    meta = dict(source_ckpt=str(ck_path), step=ck["step"], total_steps=args["steps"], cfg=ck["cfg"],
                train_npz=str(_train_only_npz().resolve()), max_changed=a.budget, recipe=recipe)
    (sess / "session.json").write_text(json.dumps(meta, indent=1))
    if not a.no_prompt:
        write_prompt(sess, meta, args)
    rows = [json.loads(l) for l in open(run / "log.jsonl") if '"loss"' in l]
    rows = [r for r in rows if r["step"] < ck["step"]]
    with open(sess / "loss_so_far.csv", "w") as f:
        f.write("step,train_loss_mean_of_10_steps,train_batch_acc_mean_of_10_steps,lr\n")
        for i in range(0, len(rows), 10):
            c = rows[i:i + 10]
            f.write(f"{c[0]['step']},{np.mean([r['loss'] for r in c]):.4f},{np.mean([r['acc'] for r in c]):.4f},"
                    f"{c[0]['lr']:.6f}\n")
    print(f"session ready at {sess} from {ck_path}")


def write_prompt(sess, meta, args):
    import os
    from model import ViT, ViTConfig
    c = meta["cfg"]
    d, H = c["d_model"], c["n_head"]
    nparams = ViT(ViTConfig(**c)).n_params()
    spe = 50000 // args["batch"]
    sess_abs = sess.resolve()
    repo = HERE.parent
    rel = lambda p: os.path.relpath(p, sess_abs).replace(os.sep, "/")
    t = (HERE / "prompt_template.md").read_text(encoding="utf-8")
    t = t.format(d=d, d2=2 * d, d3=3 * d, dm1=d - 1, d2m1=2 * d - 1, d3m1=3 * d - 1, dh=d // H, dhm1=d // H - 1,
                 H=H, L=c["n_layer"], dmlp=c["mlp_mult"] * d, nparams=nparams, wd=args["wd"], lr=args["lr"],
                 warmup=args["warmup"], minlr=f"{args['lr'] * args['min_lr_frac']:g}", total=args["steps"],
                 batch=args["batch"], spe=spe, epochs=round(args["steps"] / spe), step=meta["step"],
                 edit_epochs=round(meta["step"] / spe, 1), session_abs=str(sess_abs).replace(os.sep, "/"),
                 python_rel=rel(repo / ".venv" / "Scripts" / "python.exe"), vision_rel=rel(HERE),
                 budget=meta["max_changed"], budget_pct=round(100 * meta["max_changed"] / nparams, 1))
    (sess / "PROMPT.md").write_text(t, encoding="utf-8")


def _arms(sess):
    """Controls and continued-training arms live OUTSIDE the agent's folder (it must not see test accuracy):
    runs/vision_vit/arms/<session name>/ with a copy of session.json."""
    d = sess.parent / "arms" / sess.name
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy(sess / "session.json", d / "session.json")
    return d


def _load(sess):
    meta = json.loads((sess / "session.json").read_text())
    ck = torch.load(meta["source_ckpt"], map_location="cpu", weights_only=False)
    return meta, ck


def apply(a):
    sess = Path(a.session)
    meta, ck = _load(sess)
    src = sess / "work" / "edited" / "edit.pt"
    edit = torch.load(src, map_location="cpu") if src.exists() else {}
    for k, v in edit.items():
        assert ck["model"][k].shape == v.shape, k
        ck["model"][k] = v.to(ck["model"][k].dtype)
    out = _arms(sess) / f"edited_step_{meta['step']:05d}.pt"
    torch.save(ck, out)
    torch.save(edit, sess / "submitted_edit.pt")
    torch.save(edit, _arms(sess) / "submitted_edit.pt")
    n = sum(int((v.float() != _load_base(sess)[k]).sum()) for k, v in edit.items())
    print(f"wrote {out}: {len(edit)} tensors replaced, {n:,} entries changed: {sorted(edit)}")


def _load_base(sess):
    return {k: v.float() for k, v in torch.load(sess / "base.pt", map_location="cpu", weights_only=False)["model"].items()}


def random_edit(a):
    sess = Path(a.session)
    meta, ck = _load(sess)
    edit = torch.load(sess / "submitted_edit.pt", map_location="cpu")
    g = torch.Generator().manual_seed(a.seed)
    for k, v in edit.items():
        base = ck["model"][k].float()
        d = v.float() - base
        n = int((d != 0).sum())
        idx = torch.randperm(base.numel(), generator=g)[:n]
        r = torch.zeros(base.numel())
        r[idx] = torch.randn(n, generator=g)
        r = r.view_as(base) * (d.norm() / r.norm())
        ck["model"][k] = (base + r).to(ck["model"][k].dtype)
    out = _arms(sess) / f"random{a.seed}_step_{meta['step']:05d}.pt"
    torch.save(ck, out)
    print(f"wrote {out}")


def mimetic(a):
    from mimetic import mimetic_state
    sess = Path(a.session)
    meta, ck = _load(sess)
    new = mimetic_state(ck["model"], ck["cfg"], seed=a.seed)
    n = 0
    for k, v in new.items():
        n += int((v.float() != ck["model"][k].float()).sum())
        ck["model"][k] = v
    out = _arms(sess) / f"mimetic_step_{meta['step']:05d}.pt"
    torch.save(ck, out)
    print(f"wrote {out}: {len(new)} tensors, {n:,} entries changed")


def none(a):
    sess = Path(a.session)
    meta, _ = _load(sess)
    out = _arms(sess) / f"none_step_{meta['step']:05d}.pt"
    shutil.copy(meta["source_ckpt"], out)
    print(f"wrote {out}")


def main():
    import sys
    sys.path.insert(0, str(HERE))
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "apply", "random", "mimetic", "none"])
    ap.add_argument("--run")
    ap.add_argument("--step", type=int)
    ap.add_argument("--session", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget", type=int, default=900_000, help="prepare: max entries the agent may change")
    ap.add_argument("--no-prompt", action="store_true", help="prepare: no PROMPT.md (control-only seeds)")
    a = ap.parse_args()
    {"prepare": prepare, "apply": apply, "random": random_edit, "mimetic": mimetic, "none": none}[a.cmd](a)


if __name__ == "__main__":
    main()
