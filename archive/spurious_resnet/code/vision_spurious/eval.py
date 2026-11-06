"""Score one arm on the held-out eval split (data_vision/derived/eval.pt). Nobody fits on this split.

  python vision_spurious/eval.py --name none
  python vision_spurious/eval.py --name agent_s1 --edit runs/vision_spurious/session_1/submitted_edit.pt
  python vision_spurious/eval.py --name llr --state runs/vision_spurious/arms/llr_state.pt
  python vision_spurious/eval.py --name spufix_like --spufix runs/vision_spurious/arms/spufix_like.pt

Metrics (per group of spurious classes: dev classes on their 50 held-out images each, unseen classes on all 75):
  spur_ap    mean over classes of the official Spurious ImageNet score: average precision of p_k (softmax prob of
             class k) separating ImageNetV2 eval images of class k (7 per class, positives) from the spurious images
             of class k (negatives). Higher = less spurious. (Official uses 50 ImageNet-val positives; see plan.)
  spur_auc   same with ROC AUC
  spur_hit   fraction of spurious images whose top-1 prediction is k (lower = less spurious)
  spur_pk    mean p_k on the spurious images
  top1       ImageNetV2 matched-frequency eval top-1 (7,000 images), also on the dev / unseen spurious classes
Writes runs/vision_spurious/eval/<name>.json and <name>_logits.pt (fp16).
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def average_precision_score(y, s):
    """Same as sklearn's average_precision_score when scores have no ties."""
    o = np.argsort(-s, kind="stable")
    y = y[o]
    tp = np.cumsum(y)
    prec = tp / np.arange(1, len(y) + 1)
    return float((prec * y).sum() / y.sum())


def roc_auc_score(y, s):
    pos, neg = s[y == 1], s[y == 0]
    return float(((pos[:, None] > neg[None]).sum() + 0.5 * (pos[:, None] == neg[None]).sum()) / (len(pos) * len(neg)))


def entries_changed(state=None, edit=None):
    base = C.base_state()
    if edit is not None:
        return int(sum((v.float() != base[k].float()).sum() for k, v in edit.items()))
    if state is not None:
        return int(sum((state[k].float() != base[k].float()).sum() for k in C.editable(base)))
    return 0


def spurious_metrics(logits, labels, kind, group, classes, grp):
    probs = logits.float().softmax(1)
    pred = probs.argmax(1)
    clean = torch.tensor([k == "clean" for k in kind])
    ingrp = torch.tensor([g == grp for g in group])
    ap, auc, hit, pk = [], [], [], []
    per = {}
    for k in classes:
        pos = clean & (labels == k)
        neg = ingrp & (labels == k)
        if neg.sum() == 0:
            continue
        s = torch.cat([probs[pos, k], probs[neg, k]]).numpy()
        y = np.r_[np.ones(int(pos.sum())), np.zeros(int(neg.sum()))]
        a, u = average_precision_score(y, s), roc_auc_score(y, s)
        h = (pred[neg] == k).float().mean().item()
        p = probs[neg, k].mean().item()
        ap.append(a); auc.append(u); hit.append(h); pk.append(p)
        per[int(k)] = dict(ap=round(a, 4), auc=round(u, 4), hit=round(h, 4), pk=round(p, 4),
                           clean_top1=round((pred[pos] == k).float().mean().item(), 4))
    return dict(spur_ap=float(np.mean(ap)), spur_auc=float(np.mean(auc)), spur_hit=float(np.mean(hit)),
                spur_pk=float(np.mean(pk)), n_classes=len(ap)), per


def score(logits, d):
    labels, kind, group = d["labels"], d["kind"], d["group"]
    clean = torch.tensor([k == "clean" for k in kind])
    pred = logits.argmax(1)
    dev_c, all_c = d["dev_classes"], d["all_spurious_classes"]
    unseen_c = [k for k in all_c if k not in dev_c]
    corr = (pred == labels) & clean
    isin = lambda cs: clean & torch.isin(labels, torch.tensor(cs))  # noqa: E731
    out = dict(top1=corr.sum().item() / clean.sum().item(),
               top1_dev_classes=corr[isin(dev_c)].float().mean().item(),
               top1_unseen_classes=corr[isin(unseen_c)].float().mean().item(),
               top1_other_classes=corr[clean & ~torch.isin(labels, torch.tensor(all_c))].float().mean().item())
    m_dev, per_dev = spurious_metrics(logits, labels, kind, group, dev_c, "spur_dev_class")
    m_un, per_un = spurious_metrics(logits, labels, kind, group, unseen_c, "spur_unseen_class")
    out.update({f"dev_{k}": v for k, v in m_dev.items()})
    out.update({f"unseen_{k}": v for k, v in m_un.items()})
    return out, dict(dev=per_dev, unseen=per_un)


def spufix_head(m, path):
    """SpuFix-style clamp on top of the (unchanged) fc layer: for each class k with a direction v_k in weighted-activation
    space (w_k * phi), subtract max(0, (w_k*phi - mu_k) . v_k) * sum(v_k) from logit k (Neuhaus et al., SpuFix)."""
    S = torch.load(path, map_location="cpu")
    W, b = m.fc.weight, m.fc.bias
    ks = torch.tensor(S["classes"], device=W.device)
    V = S["v"].to(W.device)    # [K, 2048] unit vectors
    MU = S["mu"].to(W.device)  # [K, 2048] weighted-activation means

    def head(f):
        z = f @ W.T + b
        a = f[:, None, :] * W[ks][None]            # [B, K, 2048]
        beta = ((a - MU[None]) * V[None]).sum(-1)  # [B, K]
        z[:, ks] -= beta.clamp(min=0) * V.sum(-1)[None]
        return z
    return head


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--edit", help="{name: full tensor} edit applied to the base weights")
    ap.add_argument("--state", help="full replacement state dict")
    ap.add_argument("--spufix", help="SpuFix-style clamp file (classes, v, mu)")
    ap.add_argument("--bs", type=int, default=32)
    a = ap.parse_args()
    edit = torch.load(a.edit, map_location="cpu") if a.edit else None
    state = torch.load(a.state, map_location="cpu") if a.state else None
    m = C.build(state=state, edit=edit)
    head = spufix_head(m, a.spufix) if a.spufix else None
    d = torch.load(C.DERIVED / "eval.pt")
    logits, _ = C.logits_feats(m, d["images"], bs=a.bs, head=head)
    res, per = score(logits, d)
    res = dict(name=a.name, entries_changed=entries_changed(state, edit), **res)
    out = C.RUNS / "eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.name}.json").write_text(json.dumps(dict(summary=res, per_class=per), indent=1))
    torch.save(logits.half(), out / f"{a.name}_logits.pt")
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res.items()}))


if __name__ == "__main__":
    main()
