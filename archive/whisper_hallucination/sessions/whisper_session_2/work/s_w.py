import sys; sys.path.insert(0, "../../../whisper_halluc"); import agent_tools as T
import torch
W = T.base_weights()
g = W["model.decoder.layer_norm.weight"]; b = W["model.decoder.layer_norm.bias"]
print("final LN g stats", g.min().item(), g.median().item(), g.max().item(), (g.abs()<0.1).sum().item())
E = W["model.decoder.embed_tokens.weight"]; print(E.shape, E.norm(dim=1).mean().item())
for t in [50257, 1097, 2411, 291, 1044]:
    print(t, E[t].norm().item(), (b@E[t]).item())
print([k for k in W if "decoder.layers.11" in k])
print("fc2 col norm 11:", W["model.decoder.layers.11.fc2.weight"].norm(dim=0).median().item())
print("fc1 row norm 11:", W["model.decoder.layers.11.fc1.weight"].norm(dim=1).median().item(), W["model.decoder.layers.11.fc1.bias"].median().item())
