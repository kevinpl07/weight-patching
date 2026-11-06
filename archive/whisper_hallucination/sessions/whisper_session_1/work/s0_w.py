import sys; sys.path.insert(0, "../../../whisper_halluc"); import agent_tools as T
import torch
W = T.base_weights()
for k in ["model.decoder.layer_norm.weight","model.decoder.layer_norm.bias","model.decoder.layers.11.final_layer_norm.weight"]:
    v=W[k]; print(k, v.shape, float(v.min()), float(v.max()), float(v.abs().mean()))
f1=W["model.decoder.layers.11.fc1.weight"]; f2=W["model.decoder.layers.11.fc2.weight"]
print(f1.shape,f2.shape)
cn=f2.norm(dim=0); print("fc2 col norms min/med/max", float(cn.min()), float(cn.median()), float(cn.max()))
E=W["model.decoder.embed_tokens.weight"]; print("E", E.shape, float(E.norm(dim=1).mean()))
tk=T.tokenizer()
for s in [" .", ".", " ...", "...", " ♪", " -", " [", " you", " Thank"]:
    print(repr(s), tk.encode(s, add_special_tokens=False))
