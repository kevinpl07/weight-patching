import sys; sys.path.insert(0, "../../../vision_spurious"); import agent_tools as T
import torch, numpy as np
torch.set_grad_enabled(False)
d = T.dev(); W = T.base_weights(); cats = T.categories()
B = torch.load("work/base_feats.pt"); F, L0 = B["F"], B["L"]
y = d["labels"]; clean = torch.tensor([k == "clean" for k in d["kind"]]); DC = list(d["dev_classes"])
Wf, bf = W["fc.weight"].detach().clone(), W["fc.bias"].detach().clone()
def ap(lab, s):
    o = np.argsort(-s, kind="stable"); lab = lab[o]
    return float((np.cumsum(lab) / np.arange(1, len(lab)+1) * lab).sum() / lab.sum())
def offmetrics(Wn, bn, idx=None):
    Lg = F @ Wn.T + bn; P = Lg.softmax(1); pred = P.argmax(1)
    dc = torch.tensor(DC); per = {}
    for k in DC:
        pos, neg = clean & (y == k), ~clean & (y == k)
        s = torch.cat([P[pos, k], P[neg, k]]).numpy()
        lab = np.r_[np.ones(int(pos.sum())), np.zeros(int(neg.sum()))]
        per[k] = dict(hit=(pred[neg] == k).float().mean().item(), pk=P[neg, k].mean().item(), ap=ap(lab, s),
                      clean_acc=(pred[pos] == k).float().mean().item())
    res = dict(clean_top1=((pred == y) & clean).sum().item()/clean.sum().item(),
               clean_devcls=((pred == y) & clean & torch.isin(y, dc)).sum().item()/(clean & torch.isin(y, dc)).sum().item(),
               hit=np.mean([v["hit"] for v in per.values()]), pk=np.mean([v["pk"] for v in per.values()]),
               ap=np.mean([v["ap"] for v in per.values()]))
    return res, per, pred
