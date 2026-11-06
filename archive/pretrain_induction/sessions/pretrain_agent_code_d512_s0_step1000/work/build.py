"""Construct an induction circuit by hand: sinusoidal pos features in 32 coords of pos.weight,
layer-0 head H0 = previous-token head (QK from sinusoids, OV writes token code into quiet subspace Wr),
layer-1 head H1 = induction head (K reads Wr, Q/V read current-token code R, O = Hebbian code->unembed map)."""
import sys, math; sys.path.insert(0, "../../../pretrain"); import agent_tools as T
import torch, torch.nn.functional as F
W = T.base_weights(); E = W["tok.weight"]; P = W["pos.weight"]; U = W["head.weight"]
H0, H1, NQ = 6, 2, 192
_cache = {}

def stats(wr_mode="evar", c_mode="evar"):
    key = (wr_mode, c_mode)
    if key in _cache: return _cache[key]
    d = torch.load("work/r1_seed2.pt"); x = d["x"]; ho0 = d["ho0"]
    cnt = torch.bincount(T.train_windows(256, 512, seed=50).reshape(-1), minlength=8192).float()
    wf = cnt / cnt.sum()
    # read sensitivity of downstream consumers (layer-1 heads other than H1, unembedding), each trace-normalised
    Sread = torch.zeros(512, 512)
    q1 = W["blocks.1.attn.qkv.weight"]; p1 = W["blocks.1.attn.proj.weight"]
    for h in range(8):
        if h == H1: continue
        for blk in (q1[h*64:h*64+64], q1[512+h*64:512+h*64+64], p1[:, h*64:h*64+64] @ q1[1024+h*64:1024+h*64+64]):
            G = blk.T @ blk; Sread += G / G.trace()
    Upf = U * W["ln_f.weight"][None]; G = Upf.T @ Upf; Sread += 3 * G / G.trace()
    if c_mode == "evar":
        C = torch.argsort(E.var(0) + P.var(0))[:64]
    else:
        sc = (E.var(0) + P.var(0)) / (E.var(0) + P.var(0)).mean() + Sread.diagonal() / Sread.diagonal().mean()
        C = torch.argsort(sc)[:64]
    mask = torch.ones(512, dtype=torch.bool); mask[C] = False
    S0 = torch.eye(512)[mask]
    L0 = (ho0.sum(1) - ho0[:, H0]).reshape(-1, 512); L0 = L0 - L0.mean(0)
    ev, evec = torch.linalg.eigh(S0 @ (L0.T @ L0 / L0.shape[0]) @ S0.T)
    Qz = evec[:, :NQ].T @ S0
    En = F.layer_norm(E, (512,))
    Ec = En - (wf[:, None] * En).sum(0)
    e2, v2 = torch.linalg.eigh(Qz @ ((Ec.T * wf) @ Ec) @ Qz.T)
    R = v2[:, -64:].T @ Qz; Wr = v2[:, :64].T @ Qz
    if wr_mode == "read":
        Rest = v2[:, :-64].T @ Qz                      # quiet subspace minus R  (128 dims)
        CE = (Ec.T * wf) @ Ec
        Mx = Rest @ (Sread / Sread.trace() + CE / CE.trace()) @ Rest.T
        e3, v3 = torch.linalg.eigh(Mx)
        Wr = v3[:, :64].T @ Rest
    sig0 = (E[x] + P[:512]).std(-1).mean().item(); sig1 = d["r1"].std(-1).mean().item()
    _cache[key] = dict(C=C, R=R, Wr=Wr, wf=wf, En=En, sig0=sig0, sig1=sig1)
    return _cache[key]

def build(a=0.1, beta0=0.6, g0=0.25, beta1=1.0, g1=1.0, wmode="uni", nfreq=32, lo=0.5, hi=1.5, lam=0.3, pos_add=True, ln_comp=True, parts="pos,l0,l1", wr_mode="evar", c_mode="evar", merge=False, ds=1.0, blind0=0.0, kc=4):
    s = stats(wr_mode, c_mode); C, R, Wr, wf, En = s["C"], s["R"], s["Wr"], s["wf"], s["En"]
    out = {}
    # --- pos.weight: sinusoids in 32 coords
    w = torch.exp(torch.linspace(math.log(lo), math.log(hi), nfreq))
    t = torch.arange(512).float()
    Pn = P.clone()
    cc, ss = C[0:2*nfreq:2], C[1:2*nfreq:2]
    base_c = P[:, cc] if pos_add else 0; base_s = P[:, ss] if pos_add else 0
    Pn[:, cc] = base_c + a * torch.cos(t[:, None] * w[None]); Pn[:, ss] = base_s + a * torch.sin(t[:, None] * w[None])
    if "pos" in parts: out["pos.weight"] = Pn
    if ln_comp and "pos" in parts:
        # LN-0 sees a larger per-token std after adding the sinusoids; rescale ln1 gain (non-sinusoid coords) to keep
        # other layer-0 heads' inputs at their old scale on average
        x = T.train_windows(16, 512, seed=7)
        old = (E[x] + P[:512]).std(-1); new = (E[x] + Pn[:512]).std(-1)
        r = (new / old).mean().item()
        lw = W["blocks.0.ln1.weight"].clone(); lb = W["blocks.0.ln1.bias"]
        m = torch.ones(512, dtype=torch.bool); m[C] = False
        lw[m] = lw[m] * r
        out["blocks.0.ln1.weight"] = lw
        print("ln comp ratio", round(r, 4))
    # --- layer 0 prev-token head
    qkv0 = W["blocks.0.attn.qkv.weight"].clone(); lw0 = W["blocks.0.ln1.weight"]
    Q = torch.zeros(64, 512); K = torch.zeros(64, 512)
    amp = a / (s["sig0"]**2 + nfreq * a * a / 512) ** 0.5                     # typical normalized sinusoid amplitude
    qs = beta0 * 8 / amp**2
    for j in range(nfreq):
        c, sn, om = cc[j], ss[j], w[j]
        co = math.cos(om) * (1 - lam)
        Q[2*j, c] = qs * co / lw0[c]; Q[2*j, sn] = qs * math.sin(om) / lw0[sn]
        Q[2*j+1, c] = -qs * math.sin(om) / lw0[c]; Q[2*j+1, sn] = qs * co / lw0[sn]
        K[2*j, c] = 1 / lw0[c]; K[2*j+1, sn] = 1 / lw0[sn]
    sl = slice(H0*64, H0*64+64)
    f = (Q.norm() / K.norm()).sqrt()           # balance Q/K norms (same product)
    qkv0[sl] = Q / f; qkv0[512:][sl] = K * f
    qkv0[1024:][sl] = R / lw0[None]
    pr0 = W["blocks.0.attn.proj.weight"].clone(); pr0[:, sl] = g0 * Wr.T
    if merge:
        # fold the old H6 OV (a broad 'bag of context tokens' copy head) into H2, whose attention is similarly broad
        q0 = W["blocks.0.attn.qkv.weight"]; p0 = W["blocks.0.attn.proj.weight"]
        OV = sum(p0[:, h*64:h*64+64] @ q0[1024+h*64:1024+h*64+64] for h in (2, H0))
        Uo, So, Vo = torch.linalg.svd(OV)
        qkv0[1024+128:1024+192] = So[:64, None].sqrt() * Vo[:64]
        pr0[:, 128:192] = Uo[:, :64] * So[:64].sqrt()
    if blind0:
        # other layer-0 heads: scale their reads of the sinusoid coords by (1-blind0)
        for h in range(8):
            if h == H0: continue
            for base in (0, 512, 1024):
                rows = slice(base + h*64, base + h*64 + 64)
                blk = qkv0[rows]; blk[:, C] *= (1 - blind0); qkv0[rows] = blk
    if "l0" in parts: out["blocks.0.attn.qkv.weight"] = qkv0; out["blocks.0.attn.proj.weight"] = pr0
    if ds != 1.0:
        for n in ("blocks.1.ln1.weight", "ln_f.weight"):
            v = W[n].clone(); v[C] *= ds; out[n] = v
    # --- layer 1 induction head
    qkv1 = W["blocks.1.attn.qkv.weight"].clone(); lw1 = W["blocks.1.ln1.weight"]
    sl1 = slice(H1*64, H1*64+64)
    # typical code norms: code_q = R n1 where token part ~ R En * (sig_e/sig1); code_k = g0 R En /sig1 ... normalize to unit-ish
    cn = (En @ R.T).norm(dim=-1); cn = (wf * cn).sum().item()          # typical |R LN(e)|
    sig_e = E.std(-1).mean().item()
    qn = cn * sig_e / s["sig1"]                                          # typical |q code| at layer1
    kn = g0 * cn * (sig_e + 0.0) / s["sig0"] / s["sig1"] * (s["sig0"] / sig_e)  # ~ g0*cn/sig1
    kn = max(kn, 1e-6 * cn / s["sig1"]) if g0 == 0 else kn
    Q1 = beta1 * 8 / (qn * kn) * R / lw1[None]; K1 = Wr / lw1[None]
    f1 = (Q1.norm() / K1.norm()).sqrt()
    qkv1[sl1] = Q1 / f1
    qkv1[512:][sl1] = K1 * f1
    qkv1[1024:][sl1] = R / lw1[None]
    # Hebbian OV: code -> unembedding direction
    lf = out.get("ln_f.weight", W["ln_f.weight"]); Up = U * lf[None]
    ww = {"freq": wf, "sqrt": wf.sqrt(), "uni": (wf > 0).float()}[wmode]; ww = ww / ww.sum()
    Upc = Up - (ww[:, None] * Up).sum(0)
    codes = En @ R.T                       # [V,64] code of clean token
    codes = codes - (ww[:, None] * codes).sum(0)
    if kc:
        # project out the top-kc principal directions of the token codes (shared, non-identity-specific
        # components that make every copied token look alike)
        ec, vc = torch.linalg.eigh((codes.T * ww) @ codes)
        Pc = torch.eye(64) - vc[:, 64-kc:] @ vc[:, 64-kc:].T
    else:
        Pc = torch.eye(64)
    M = (Upc.T * ww) @ codes @ Pc            # [512,64]
    M = M - M.mean(0, keepdim=True)
    # calibrate: logit boost of token b for its clean code (with value scale qn) ~ g1 nats on average
    vb = codes * (qn / cn)
    sigf = 0.36
    lg = (vb @ M.T) / sigf                 # residual contribution -> logits via Upc
    boost = ((lg * Upc).sum(-1) - 0)       # logit of own token relative to weighted mean
    b = (ww * boost).sum().item()
    pr1 = W["blocks.1.attn.proj.weight"].clone(); pr1[:, sl1] = g1 * M / b
    if "l1" in parts: out["blocks.1.attn.qkv.weight"] = qkv1; out["blocks.1.attn.proj.weight"] = pr1
    return out

def copy_check(wmode="sqrt"):
    s = stats(); o = build(wmode=wmode)
    lf = out.get("ln_f.weight", W["ln_f.weight"]); Up = U * lf[None]
    M = o["blocks.1.attn.proj.weight"][:, H1*64:H1*64+64]
    top = torch.argsort(s["wf"], descending=True)[:2000]
    codes = s["En"][top] @ s["R"].T
    lg = (codes @ M.T) @ Up.T                      # [n,V]
    lg = lg - lg.mean(-1, keepdim=True)
    own = lg[torch.arange(len(top)), top]
    rank = (lg > own[:, None]).sum(-1).float()
    print(wmode, "copy: own-token rank median", rank.median().item(), "top1", (rank == 0).float().mean().item(), "top10", (rank < 10).float().mean().item(),
          "own z-score", (own / lg.std(-1)).mean().item())

if __name__ == "__main__":
    for m in ("freq", "sqrt", "uni"): copy_check(m)
