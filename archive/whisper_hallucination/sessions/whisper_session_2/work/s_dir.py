import sys; sys.path.insert(0, "../../../whisper_halluc"); import agent_tools as T
import torch, numpy as np
D = torch.load("work/collect_snap.pt"); W = T.base_weights(); tk = T.tokenizer()
E = W["model.decoder.embed_tokens.weight"]; g = W["model.decoder.layer_norm.weight"]
comp = sorted({t for i in D if i.startswith("pns_") for t in D[i]["topk_ids"][3].tolist()} - {2411, 50257})
print(len(comp), tk.convert_ids_to_tokens(comp[:30]))
Ec = E[comp].mean(0)
for name, w in [("dot", E[2411]-Ec), ("dot+eot", E[2411]-Ec + 0.5*(E[50257]-Ec))]:
    v = w / g; v = v - v.mean(); v = v / v.norm()
    c = (g * v) @ E.T
    other = torch.cat([c[:2411], c[2412:50257]])
    top = other.topk(5)
    print(name, "dot", round(float(c[2411]),3), "eot", round(float(c[50257]),3), "you", round(float(c[291]),3), "I", round(float(c[286]),3),
          "max other", round(float(top.values[0]),3), tk.convert_ids_to_tokens(top.indices.tolist()), "mean comp", round(float(c[comp].mean()),3))
