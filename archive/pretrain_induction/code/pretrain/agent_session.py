"""Set up an agent session from a pretraining checkpoint, and apply its edit afterwards.

  prepare: python pretrain/agent_session.py prepare --run runs/pretrain_code/attn2L_d512_s0 --step 1000 \
               --session runs/pretrain_agent/code_d512_s0_step1000
  apply:   python pretrain/agent_session.py apply --session runs/pretrain_agent/code_d512_s0_step1000
           -> <session>/edited_step_XXXXX.pt: the full original checkpoint (with optimizer state) with the
              agent's tensors written in; resume it with train.py --init.
  random:  python pretrain/agent_session.py random --session ... --seed 0
           -> <session>/random{seed}_step_XXXXX.pt: same tensors, same number of changed entries per tensor
              and same L2 norm of the change per tensor, at random positions with random directions.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch


def prepare(a):
    run, sess = Path(a.run), Path(a.session)
    ck_path = run / "ckpt" / f"step_{a.step:05d}.pt"
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    sess.mkdir(parents=True, exist_ok=True)
    (sess / "work").mkdir(exist_ok=True)
    torch.save({"model": ck["model"], "cfg": ck["cfg"], "step": ck["step"]}, sess / "base.pt")
    args = ck["args"]
    recipe = dict(optimizer="AdamW betas (0.9, 0.95), weight decay 0.1 on matrices, grad clip 1.0",
                  lr=f"peak {args['lr']}, linear warmup {args['warmup']} steps, cosine decay to "
                     f"{args['min_lr_frac']} x peak at step {args['steps']}",
                  batch=f"{args['batch']} sequences x {ck['cfg']['ctx']} tokens per step",
                  data="Python source files (codeparrot-clean), byte-level BPE with 8192 tokens, files joined by <|eos|>=0"
                  if "code" in str(args["data_dir"]) else str(args["data_dir"]),
                  note="after your edit, training resumes from this step with the same optimizer state and the "
                       "same batch sequence, until total_steps")
    meta = dict(source_ckpt=str(ck_path), step=ck["step"], total_steps=args["steps"], cfg=ck["cfg"],
                data_dir=str(Path(args["data_dir"]).resolve()), recipe=recipe)
    (sess / "session.json").write_text(json.dumps(meta, indent=1))
    # training loss so far, averaged over 25-step bins (earlier segments' logs first, later ones override)
    by_step = {}
    for lp in [*(a.history_log or []), run / "log.jsonl"]:
        for l in open(lp):
            if '"loss"' in l:
                r = json.loads(l)
                by_step[r["step"]] = r
    rows = [by_step[k] for k in sorted(by_step) if k < ck["step"]]
    for i, n in enumerate(a.prev_notes or []):
        shutil.copy(n, sess / f"previous_notes_{i + 1}.md")
    with open(sess / "loss_so_far.csv", "w") as f:
        f.write("step,train_loss_mean_of_25_steps\n")
        for i in range(0, len(rows), 25):
            chunk = rows[i:i + 25]
            f.write(f"{chunk[0]['step']},{sum(r['loss'] for r in chunk) / len(chunk):.4f}\n")
    print(f"session ready at {sess} from {ck_path}")


def _load(sess):
    meta = json.loads((sess / "session.json").read_text())
    ck = torch.load(meta["source_ckpt"], map_location="cpu", weights_only=False)
    return meta, ck


def apply(a):
    sess = Path(a.session)
    meta, ck = _load(sess)
    edit = torch.load(sess / "work" / "edited" / "edit.pt", map_location="cpu")
    for k, v in edit.items():
        assert ck["model"][k].shape == v.shape, k
        ck["model"][k] = v.to(ck["model"][k].dtype)
    out = sess / f"edited_step_{meta['step']:05d}.pt"
    torch.save(ck, out)
    shutil.copy(sess / "work" / "edited" / "edit.pt", sess / "submitted_edit.pt")
    print(f"wrote {out}: {len(edit)} tensors replaced: {sorted(edit)}")


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
    out = sess / f"random{a.seed}_step_{meta['step']:05d}.pt"
    torch.save(ck, out)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "apply", "random"])
    ap.add_argument("--run")
    ap.add_argument("--step", type=int)
    ap.add_argument("--session", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--history-log", nargs="*", help="prepare: earlier segments' log.jsonl files for loss_so_far.csv")
    ap.add_argument("--prev-notes", nargs="*", help="prepare: earlier sessions' notes.md, copied in as previous_notes_N.md")
    a = ap.parse_args()
    {"prepare": prepare, "apply": apply, "random": random_edit}[a.cmd](a)


if __name__ == "__main__":
    main()
