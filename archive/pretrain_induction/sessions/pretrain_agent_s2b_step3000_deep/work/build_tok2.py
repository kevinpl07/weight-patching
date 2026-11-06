"""Second token-code head in L0 (self-attention, identity code complementary to L0H4) + optional second copy
head in L1 (attention = L1H2's QK, OV = Hebbian map from the new code to the unembedding)."""
import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch, torch.nn.functional as F, json
torch.manual_seed(0)
W = T.base_weights()
d = 512; V = 8192
H0new, H1new = int(sys.argv[1]) if len(sys.argv)>1 else 5, int(sys.argv[2]) if len(sys.argv)>2 else 5
g0 = float(sys.argv[3]) if len(sys.argv)>3 else 6.0     # output norm of new L0 head
g1 = float(sys.argv[4]) if len(sys.argv)>4 else 0.0     # own-token logit boost of new L1 head (0 = no L1 head)
wexp = float(sys.argv[5]) if len(sys.argv)>5 else 0.5   # freq weighting exponent
mode = sys.argv[6] if len(sys.argv)>6 else "save"
# token frequencies
xs = T.train_windows(2000, 512, seed=7000)
freq = torch.bincount(xs.reshape(-1), minlength=V).float() + 1
freq = freq / freq.sum()
tokW = W["tok.weight"]; pos = W["pos.weight"]
ln0w, ln0b = W["blocks.0.ln1.weight"], W["blocks.0.ln1.bias"]
qkv0 = W["blocks.0.attn.qkv.weight"].clone(); P0 = W["blocks.0.attn.proj.weight"].clone()
qkv1 = W["blocks.1.attn.qkv.weight"].clone(); P1 = W["blocks.1.attn.proj.weight"].clone()
ln1w, ln1b = W["blocks.1.ln1.weight"], W["blocks.1.ln1.bias"]
Z = F.layer_norm(tokW + pos.mean(0), (d,), ln0w, ln0b)          # [V, d] layer-0 input per token
wt = freq ** wexp; wt = wt / wt.sum()
mu = (wt[:, None] * Z).sum(0)
Zc = Z - mu
Wv4 = qkv0[1024 + 4*64: 1024 + 5*64]                              # L0H4 V rows
Q4, _ = torch.linalg.qr(Wv4.T)                                    # [d,64] row space of L0H4 V
Zp = Zc - (Zc @ Q4) @ Q4.T
C = (Zp * wt[:, None]).T @ Zp
ev, evec = torch.linalg.eigh(C)
U = evec[:, -64:].flip(1)                                          # top-64 directions [d,64]
print("variance captured by top64 of complement: %.3f, L0H4-space share of total: %.3f" % (
    ev[-64:].sum() / ev.sum(), 1 - (Zp * Zp * wt[:, None]).sum() / (Zc * Zc * wt[:, None]).sum()))
# scale: whiten the code so all 64 dims have unit variance (identity code), so v = Uw^T z
code_sd = ev[-64:].flip(0).sqrt()
Vrows = (U / code_sd).T                                          # [64, d] -> code with unit variance per dim
# Q,K: self attention via L0H6's K rows (sinusoid position code)
K6 = qkv0[512 + 6*64: 512 + 7*64]
# residual write directions D: quiet in resid1/resid2, small logit gain, small L1 read, orthogonal to ones
x = T.train_windows(8, 512, seed=31)
o = T.run(x, edited=False, resid=True)
R1 = o["resid"][1].reshape(-1, d); R2 = o["resid"][2].reshape(-1, d)
def cov(R):
    Rc = R - R.mean(0); Rc = Rc - Rc.mean(1, keepdim=True); c = Rc.T @ Rc / len(Rc); return c / torch.trace(c)
lnfw = W["ln_f.weight"]; Wu = W["head.weight"]
G = (Wu * lnfw).T; G = G - G.mean(1, keepdim=True); Gm = G @ G.T; Gm = Gm / torch.trace(Gm)
L1r = qkv1 * ln1w; Lm = L1r.T @ L1r; Lm = Lm / torch.trace(Lm)
M = cov(R1) + cov(R2) + Gm + Lm
one = torch.ones(d, 1) / d ** 0.5
Pp = torch.eye(d) - one @ one.T
M = Pp @ M @ Pp + 10 * one @ one.T
evm, evecm = torch.linalg.eigh(M)
D = evecm[:, :64]                                                  # [d, 64] quietest
print("D quietness (mean eig / mean all): %.4f" % (evm[:64].mean() / evm.mean()))
qkv0n = qkv0.clone(); P0n = P0.clone()
sl = slice(H0new*64, H0new*64+64)
qkv0n[0:512][sl] = K6; qkv0n[512:1024][sl] = K6
qkv0n[1024:1536][sl] = Vrows
# O: output = g0 * D code / sqrt(64) so that |out| ~ g0 (code has unit var per dim -> |code| ~ 8)
P0n[:, sl] = D * (g0 / 8.0)
edit = {"blocks.0.attn.qkv.weight": qkv0n, "blocks.0.attn.proj.weight": P0n}
meta = dict(H0new=H0new, H1new=H1new, g0=g0, g1=g1, wexp=wexp)
if g1 > 0:
    # L1 copy head: Q,K = L1H2; V reads D (undo ln1 weight); O = Hebbian map code -> unembedding
    s2 = slice(H1new*64, H1new*64+64)
    qkv1n = qkv1.clone(); P1n = P1.clone()
    qkv1n[0:512][s2] = qkv1[0:512][2*64:3*64]; qkv1n[512:1024][s2] = qkv1[512:1024][2*64:3*64]
    sig1 = (R1 - R1.mean(1, keepdim=True)).pow(2).mean(1).sqrt().mean()   # typical LN sigma at layer 1
    Vr1 = (D / ln1w[:, None]).T * (sig1 / (g0 / 8.0))                      # value = code (unit var per dim)
    qkv1n[1024:1536][s2] = Vr1
    codes = (Zc @ U) / code_sd                                             # [V, 64] codes of tokens (unit var)
    sig2 = (R2 - R2.mean(1, keepdim=True)).pow(2).mean(1).sqrt().mean()
    Uemb = (Wu * lnfw); Uemb = Uemb - Uemb.mean(0)                         # [V, d] logit read of a resid dir (x 1/sig2)
    wts = freq ** wexp; wts = wts / wts.sum()
    Oh = (Uemb * wts[:, None]).T @ codes                                     # [d, 64] Hebbian
    # project Oh off the all-ones direction
    Oh = Oh - Oh.mean(0, keepdim=True)
    # measure own boost: logits = Uemb @ (Oh @ code_b) / sig2
    idx = torch.multinomial(freq, 2000, replacement=True)
    lg = (Uemb @ (Oh @ codes[idx].T)).T / sig2                               # [2000, V]
    own = lg.gather(1, idx[:, None]).squeeze(1)
    z = (own - lg.mean(1)) / lg.std(1)
    rank = (lg > own[:, None]).sum(1).float()
    scale = g1 / (own - lg.mean(1)).mean()
    print("Hebbian copy: own z %.2f, median rank %d, top1 %.2f; scale %.3g" % (z.mean(), rank.median(), (rank == 0).float().mean(), scale))
    P1n[:, s2] = Oh * scale
    edit.update({"blocks.1.attn.qkv.weight": qkv1n, "blocks.1.attn.proj.weight": P1n})
if mode == "save":
    T.save_edit(edit, note=json.dumps(meta))
torch.save(dict(U=U, D=D, code_sd=code_sd), "work/tok2_basis.pt")
