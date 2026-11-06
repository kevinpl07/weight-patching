import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch
def build(r, specs):
    """r: unit direction [3072]. specs: list of (param name, alpha). Output matrices: W' = W - alpha * r r^T W."""
    W = T.base_weights(); out = {}
    r = r.float() / r.float().norm()
    for name, a in specs:
        w = W[name].float()
        out[name] = w - a * torch.outer(r, r @ w)
    return out
