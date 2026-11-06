import torch
def make_random(W, ed, seed=0):
    """Random control matching the real edit: same tensors, same number of changed columns per tensor, same
    Frobenius norm of the change per tensor, but random columns and a random direction."""
    g = torch.Generator().manual_seed(seed)
    out = {}
    for name, new in ed.items():
        base = W[name].float(); delta = new.float() - base
        ncol = int((delta.abs().sum(0) > 0).sum())
        fro = delta.norm()
        cols = torch.randperm(base.shape[1], generator=g)[:ncol]
        r = torch.randn(base.shape[0], generator=g); r /= r.norm()
        dl = torch.zeros_like(base)
        dl[:, cols] = -r[:, None] * (r @ base[:, cols])[None, :]
        dl *= fro / dl.norm()
        out[name] = (base + dl).to(torch.bfloat16)
    return out
