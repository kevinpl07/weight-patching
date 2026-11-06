import sys; sys.path.insert(0, "../../../whisper_halluc"); import agent_tools as T
import torch, numpy as np
D = torch.load("work/collect_snap.pt"); W = T.base_weights()
E = W["model.decoder.embed_tokens.weight"]; g = W["model.decoder.layer_norm.weight"]
tgt = E[2411] + E[50257] - 2 * E[:50257].mean(0)
v = tgt / g; v = v - v.mean(); v = v / v.norm()
c = (g * v) @ E.T
print("per-unit logit change: ' .'", float(c[2411]), "eot", float(c[50257]), "' you'", float(c[291]), "' I'", float(c[286]), "median all", float(c.median()), "max other", float(torch.cat([c[:2411], c[2412:50257]]).max()))
sig = [float(D[i]["hidden"][11][3].float().std()) for i in D if i.startswith("pns_")]
print("std of residual entering layer 11 at pos3 (ns):", np.percentile(sig, [10, 50, 90]))
