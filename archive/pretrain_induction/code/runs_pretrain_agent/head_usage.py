"""Head usage on a trained 2-layer attention-only checkpoint: attention distance, ablation cost, pairwise redundancy."""
import sys, itertools
import numpy as np, torch, torch.nn.functional as F
sys.path.insert(0, "pretrain")
from model import GPT, GPTConfig
ck = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
m = GPT(GPTConfig(**ck["cfg"])); m.load_state_dict(ck["model"]); m = m.cuda().eval()
arr = np.memmap("pretrain/data_code/val.bin", dtype=np.uint16, mode="r")
rng = np.random.default_rng(0); off = rng.integers(0, len(arr) - 513, 48)
x = torch.tensor(np.stack([arr[o:o + 513] for o in off]).astype(np.int64)).cuda()
B, T = 48, 512

@torch.no_grad()
def fwd(mask=None, keep=False):
    h = m.tok(x[:, :-1]) + m.pos(torch.arange(T, device="cuda"))
    pats, outs = [], []
    for l, blk in enumerate(m.blocks):
        a = blk.attn; z = blk.ln1(h)
        q, k, v = a.qkv(z).view(B, T, 3, a.h, a.dh).permute(2, 0, 3, 1, 4)
        s = (q @ k.transpose(-1, -2)) / a.dh ** 0.5
        s = s.masked_fill(~torch.ones(T, T, dtype=torch.bool, device="cuda").tril(), float("-inf"))
        p = s.softmax(-1); y = p @ v
        Wo = a.proj.weight.view(a.proj.weight.shape[0], a.h, a.dh)
        ho = torch.einsum("bhtk,dhk->bhtd", y, Wo)
        if mask is not None:
            for (ml, mh) in mask:
                if ml == l: ho[:, mh] = 0
        h = h + ho.sum(1)
        if keep: pats.append(p); outs.append(ho)
    lg = m.head(m.ln_f(h))
    loss = F.cross_entropy(lg.transpose(1, 2), x[:, 1:]).item()
    return (loss, pats, outs) if keep else loss

base, pats, outs = fwd(keep=True)
dist = torch.arange(T, device="cuda")[:, None] - torch.arange(T, device="cuda")[None]
print(f"base val loss {base:.4f}")
print("head   mean_dist  within_4  entropy  ablate_cost  out_norm")
cost = {}
for l in range(2):
    for hh in range(8):
        p = pats[l][:, hh]
        md = (p * dist).sum(-1)[:, 64:].mean().item()
        w4 = (p * (dist <= 4)).sum(-1)[:, 64:].mean().item()
        ent = -(p.clamp_min(1e-9).log() * p).sum(-1)[:, 64:].mean().item()
        c = fwd([(l, hh)]) - base; cost[(l, hh)] = c
        print(f"L{l}H{hh}   {md:8.1f}  {w4:8.2f}  {ent:7.2f}  {c:+10.4f}  {outs[l][:, hh].norm(dim=-1).mean().item():7.2f}")
print(f"sum of single ablation costs: L0 {sum(cost[(0,h)] for h in range(8)):.3f}  L1 {sum(cost[(1,h)] for h in range(8)):.3f}")
print("pairs (same layer): joint cost vs sum of singles; ratio < 1 = overlapping (redundant), > 1 = complementary")
rows = []
for l in range(2):
    for a_, b_ in itertools.combinations(range(8), 2):
        j = fwd([(l, a_), (l, b_)]) - base
        s_ = cost[(l, a_)] + cost[(l, b_)]
        cos = F.cosine_similarity(outs[l][:, a_].reshape(-1), outs[l][:, b_].reshape(-1), dim=0).item()
        rows.append((j / s_, l, a_, b_, j, s_, cos))
rows.sort()
for r in rows[:6]: print(f"  most redundant  L{r[1]} H{r[2]}+H{r[3]}: joint {r[4]:.3f} vs sum {r[5]:.3f} (ratio {r[0]:.2f}), output cos {r[6]:+.2f}")
for r in rows[-3:]: print(f"  most complementary L{r[1]} H{r[2]}+H{r[3]}: joint {r[4]:.3f} vs sum {r[5]:.3f} (ratio {r[0]:.2f})")
print(f"median pair ratio {np.median([r[0] for r in rows]):.2f}")
