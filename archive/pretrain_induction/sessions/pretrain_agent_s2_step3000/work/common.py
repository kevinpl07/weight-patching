import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch
W = T.base_weights()
def clone(): return {k: v.clone() for k, v in W.items() if k not in ("tok.weight","head.weight")}
def qkv_rows(part, h):  # part 0=Q 1=K 2=V
    return slice(part*512 + h*64, part*512 + h*64 + 64)
def head_cols(h): return slice(h*64, h*64+64)
def eval_edit(ed, x, note="probe", attn=False):
    """ed: dict of changed tensors; saves and runs edited"""
    T.save_edit(ed, note=note)
    return T.run(x, edited=True, attn=attn)
def cats(x):
    """category per position (B, L-1): 0 nomatch 1 m1_right 2 m1_wrong 3 m2_right 4 m2_wrong"""
    B,L = x.shape
    C = torch.zeros(B, L-1, dtype=torch.long)
    for b in range(B):
        seq = x[b].tolist(); last = {}; last2 = {}
        for t in range(L-1):
            cur = seq[t]
            if cur in last:
                k2 = (seq[t-1], cur) if t>0 else None
                if k2 is not None and k2 in last2:
                    s = last2[k2]; C[b,t] = 3 if seq[s+1]==seq[t+1] else 4
                else:
                    s = last[cur]; C[b,t] = 1 if seq[s+1]==seq[t+1] else 2
            last[cur] = t
            if t>0: last2[(seq[t-1],cur)] = t
    return C
def report(loss, C, tag):
    s = f"{tag}: all {loss.mean():.4f} |"
    for i,n in enumerate(["nm","m1r","m1w","m2r","m2w"]):
        s += f" {n} {loss[C==i].mean():.3f}"
    pos = loss.mean(0)
    s += " | pos " + " ".join(f"{pos[a:b].mean():.3f}" for a,b in [(0,64),(64,128),(128,256),(256,511)])
    print(s, flush=True)
