import sys; sys.path.insert(0, "../../../vision_spurious"); import agent_tools as T
import torch, json
s = T.info()
d = T.dev()
F, L = [], []
N = len(d["labels"])
for i in range(0, N, 64):
    o = T.run(d["images"][i:i+64], edited=False, layers=["avgpool"])
    F.append(o["feats"]["avgpool"]); L.append(o["logits"])
F = torch.cat(F); L = torch.cat(L)
torch.save(dict(F=F, L=L), "work/base_feats.pt")
W = T.base_weights()
# sanity: logits from fc
print("fc recon err", (F @ W["fc.weight"].T + W["fc.bias"] - L).abs().max().item())
T.budget()
