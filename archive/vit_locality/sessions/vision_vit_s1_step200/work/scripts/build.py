import sys; sys.path.insert(0, "../../../vision"); import agent_tools as T
import torch, json, math
W = T.base_weights()
D = 192
r = torch.arange(64)//8; c = torch.arange(64)%8
rr = (r.float()-3.5); cc = (c.float()-3.5)
s = rr**2+cc**2; s_t = s/12.25; s_t = s_t - s_t.mean()

def directions(x0):
    # x0: embed resid [B,64,D]; normalized content covariance; choose 3 lowest-variance dirs orthogonal to ones
    n = torch.nn.functional.layer_norm(x0, (D,))
    n = n.reshape(-1, D); n = n - n.mean(0)
    C = n.T@n/len(n)
    one = torch.ones(D,1)/math.sqrt(D)
    Pm = torch.eye(D) - one@one.T
    C = Pm@C@Pm + 10*one@one.T
    ev, V = torch.linalg.eigh(C)
    return V[:, :3], ev[:3]

def build(A=1.0, sigma=1.0, layers=(0,1), offsets=None, dirs=None, kappa=None, ndims=3, heads=None):
    """offsets: dict layer -> list of 3 (dr, dc) per head; None means centered."""
    new = {}
    d1, d2, d3 = dirs[:,0], dirs[:,1], dirs[:,2]
    pe = W["pos_embed"].clone()
    pe[0] += A*((rr/3.5)[:,None]*d1 + (cc/3.5)[:,None]*d2 + s_t[:,None]*d3)
    new["pos_embed"] = pe
    for l in layers:
        qkw = W[f"blocks.{l}.attn.qkv.weight"].clone(); qkb = W[f"blocks.{l}.attn.qkv.bias"].clone()
        g = W[f"blocks.{l}.ln1.weight"]; b = W[f"blocks.{l}.ln1.bias"]
        k_l = kappa[l]
        lam = -k_l*A/2
        for h in range(3):
            if heads is not None and h not in heads.get(str(l), heads.get(l, [0,1,2])): continue
            sg = sigma[str(l)][h] if isinstance(sigma, dict) else sigma
            ab = 98.0/(sg**2 * k_l**2 * A**2)
            a = math.sqrt(ab); bb = math.sqrt(ab)
            dr, dc = (offsets or {}).get(str(l), [(0,0)]*3)[h]
            qr = qkw[h*64:(h+1)*64]; kr = qkw[D+h*64:D+(h+1)*64]
            score = qr.norm(dim=1)*kr.norm(dim=1)
            idx = score.argsort()[:3].tolist()
            readers = [d1/g, d2/g, d3/g]
            # Q dims: (f1, f2, 0) + bias (offset terms, 1)
            qvals = [(a*readers[0], a*(k_l*A*dr/3.5)), (a*readers[1], a*(k_l*A*dc/3.5)), (torch.zeros(D), a*1.0)]
            kvals = [(bb*readers[0], 0.0), (bb*readers[1], 0.0), (bb*lam*readers[2], 0.0)]
            for j, i in enumerate(idx):
                wq, bq = qvals[j]; wk, bk = kvals[j]
                qkw[h*64+i] = wq; qkb[h*64+i] = bq - (wq*b).sum()
                qkw[D+h*64+i] = wk; qkb[D+h*64+i] = bk - (wk*b).sum()
        new[f"blocks.{l}.attn.qkv.weight"] = qkw; new[f"blocks.{l}.attn.qkv.bias"] = qkb
    return new
