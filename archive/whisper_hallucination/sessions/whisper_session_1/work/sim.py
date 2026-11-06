"""Offline exact simulation of a layer-11 single-neuron edit using captured base internals (layer 11 is the last
layer, so its MLP output at a position only reaches the final LN + unembedding at that same position)."""
import sys; sys.path.insert(0, "../../../whisper_halluc"); import agent_tools as T
import torch, numpy as np, math
torch.set_grad_enabled(False)
W = T.base_weights()
X = "model.decoder.layers.11.final_layer_norm"; F1="model.decoder.layers.11.fc1"; H="model.decoder.layers.11"
E = W["model.decoder.embed_tokens.weight"]; g = W["model.decoder.layer_norm.weight"]; b = W["model.decoder.layer_norm.bias"]
f1w = W["model.decoder.layers.11.fc1.weight"]; f1b = W["model.decoder.layers.11.fc1.bias"]; f2w = W["model.decoder.layers.11.fc2.weight"]
_tok = None
def tok():
    global _tok
    if _tok is None: _tok = T._tok(T._get_model(False))
    return _tok
EOT = 50257; DOT = 2411
def logits_from_h(h):
    return torch.nn.functional.layer_norm(h, (768,), g, b, 1e-5) @ E.T
def mask(lg, first):
    tk = tok(); lg = lg.clone(); lg[..., tk.suppress] = -math.inf
    if first: lg[..., tk.begin_suppress] = -math.inf
    return lg
def make_edit(n, du, a, t, v):
    """returns new (f1w, f1b, f2w) with neuron n repurposed: pre = a*(x.du - t); out col = v"""
    w1 = f1w.clone(); bb = f1b.clone(); w2 = f2w.clone()
    w1[n] = a * du; bb[n] = -a * t; w2[:, n] = v
    return w1, bb, w2
def delta(rec_mods, n, new):
    x = rec_mods[X].float(); pre_old = rec_mods[F1].float()[:, n]
    w1, bb, w2 = new
    pre_new = x @ w1[n] + bb[n]
    act_new = torch.nn.functional.gelu(pre_new); act_old = torch.nn.functional.gelu(pre_old)
    return act_new[:, None] * w2[:, n][None] - act_old[:, None] * f2w[:, n][None], act_new
def sim_clip(r, n, new, start=3, first_pos=3):
    """For each position i>=start predicting seq[i+1]: base argmax and new argmax (masked)."""
    h = r["mods"][H].float(); L = h.shape[0]
    d, act = delta(r["mods"], n, new)
    lb = logits_from_h(h[start:L]); ln = logits_from_h(h[start:L] + d[start:L])
    res = []
    for j, i in enumerate(range(start, L)):
        first = (i == first_pos)
        ab = int(mask(lb[j], first).argmax()); an = int(mask(ln[j], first).argmax())
        res.append((i, ab, an, float(act[i])))
    return res
