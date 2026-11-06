import sys; sys.path.insert(0, "../../../vision"); sys.path.insert(0, "work/scripts"); import agent_tools as T
import torch, math
W = T.base_weights()
D = 192
def coords_xy():
    t = torch.arange(64); return (t % 8).float(), (t // 8).float()

def build(c=1.0, A=(6.5, 3.0, 1.5), layers=(0, 1, 2, 3), s_layer=None, feat_coords=None, clean=True, w=math.pi/8):
    """Locality edit. feat_coords: 4 residual coordinates for [cos wx, sin wx, cos wy, sin wy].
    For each block in layers and each head h, 4 Q dims and 4 K dims are replaced by readouts of those
    coordinates, scaled so the positional score is ~A[h]*(cos(w dx)+cos(w dy))."""
    E = {}
    x, y = coords_xy()
    phi = torch.stack([torch.cos(w*x), torch.sin(w*x), torch.cos(w*y), torch.sin(w*y)], 1)  # [64,4]
    pe = W['pos_embed'].clone()
    for i, a in enumerate(feat_coords):
        pe[0, :, a] = c * phi[:, i]
    E['pos_embed'] = pe
    fc = torch.tensor(feat_coords)
    for l in range(6):
        qkv = W[f'blocks.{l}.attn.qkv.weight'].clone(); qb = W[f'blocks.{l}.attn.qkv.bias'].clone()
        if clean:
            # stop other weights reading/writing the feature coordinates
            qkv[:, fc] = 0
            fc1 = W[f'blocks.{l}.mlp.fc1.weight'].clone(); fc1[:, fc] = 0; E[f'blocks.{l}.mlp.fc1.weight'] = fc1
            pw = W[f'blocks.{l}.attn.proj.weight'].clone(); pw[fc, :] = 0; E[f'blocks.{l}.attn.proj.weight'] = pw
            pb = W[f'blocks.{l}.attn.proj.bias'].clone(); pb[fc] = 0; E[f'blocks.{l}.attn.proj.bias'] = pb
            f2 = W[f'blocks.{l}.mlp.fc2.weight'].clone(); f2[fc, :] = 0; E[f'blocks.{l}.mlp.fc2.weight'] = f2
            f2b = W[f'blocks.{l}.mlp.fc2.bias'].clone(); f2b[fc] = 0; E[f'blocks.{l}.mlp.fc2.bias'] = f2b
        if l in layers:
            g = W[f'blocks.{l}.ln1.weight'][fc]; b = W[f'blocks.{l}.ln1.bias'][fc]
            for h in range(3):
                Wq = qkv[h*64:(h+1)*64]; Wk = qkv[192+h*64:192+(h+1)*64]
                imp = W[f'blocks.{l}.attn.qkv.weight'][h*64:(h+1)*64].norm(dim=1) * W[f'blocks.{l}.attn.qkv.weight'][192+h*64:192+(h+1)*64].norm(dim=1)
                dims = imp.argsort()[:4]
                beta = math.sqrt(8 * A[h]) * s_layer[l] / c   # q_m = beta * (h_a/s)
                for i, m in enumerate(dims.tolist()):
                    for off in (h*64, 192+h*64):
                        r = off + m
                        qkv[r] = 0; qkv[r, fc[i]] = beta / g[i]; qb[r] = -beta * b[i] / g[i]
        E[f'blocks.{l}.attn.qkv.weight'] = qkv; E[f'blocks.{l}.attn.qkv.bias'] = qb
    if clean:
        nw = W['norm.weight'].clone(); nw[fc] = 0; nb = W['norm.bias'].clone(); nb[fc] = 0
        E['norm.weight'] = nw; E['norm.bias'] = nb
    return E

def evaluate(tag):
    m = T.metrics(edited=True)
    print(tag, {k: m[k] for k in ['loss', 'acc', 'acc_aug']})
    print('  per_class', m['per_class_acc'])
    for l in range(6): print('  L', l, 'dist', m['dist'][l], 'local', m['local'][l])
    return m
