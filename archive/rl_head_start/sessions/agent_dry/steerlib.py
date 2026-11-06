import torch
# (layer, near-constant neuron) used as a carrier for a constant residual write
CARRIERS = {15: 1184, 16: 3, 19: 16, 21: 0}
CARRIER_MEAN = {15: -0.661, 16: 0.418, 19: 0.485, 21: 2.743}
def make_steer(W, beta, layers=(15, 16, 19, 21), direction_layer=None):
    """Add -beta/len(layers) * m_L (problem-matched mean residual difference, truncated-minus-finished answers)
    at each layer L, written through the down_proj column of a near-constant MLP neuron."""
    res = torch.load("work/seqlevel.pt", weights_only=False)
    out = {}
    for L in layers:
        m = res[("hmean", direction_layer or L)][0].float()
        n = CARRIERS[L]; a = CARRIER_MEAN[L]
        name = f"model.layers.{L}.mlp.down_proj.weight"
        Wd = W[name].float().clone()
        Wd[:, n] += -(beta / len(layers)) * m / a
        out[name] = Wd.to(torch.bfloat16)
    return out
