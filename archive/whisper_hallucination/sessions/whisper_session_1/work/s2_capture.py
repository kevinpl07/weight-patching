import sys; sys.path.insert(0, "../../../whisper_halluc"); import agent_tools as T
import torch
P = T.practice()
mods = ["model.decoder.layers.11.final_layer_norm", "model.decoder.layers.11.fc1", "model.decoder.layers.11", "model.decoder.layer_norm"]
idx = list(range(len(P["ids"])))
chunks = [idx[0:64], idx[64:100], idx[100:164], idx[164:228], idx[228:300]]
# 228:300 is 72 -> split
chunks = [idx[0:64], idx[64:100], idx[100:164], idx[164:200], idx[200:250], idx[250:300]]
allr = []
for ch in chunks:
    out = T.run(ch, edited=False, modules=mods, topk=10, max_new_tokens=443)
    for r in out:
        r2 = {k: r[k] for k in ("id","text","norm_text","tokens","seq","nsp","avg_logprob","p_eot_first","topk_ids","topk_logprobs")}
        r2["mods"] = {k: v.half() for k, v in r["modules"].items()}
        allr.append(r2)
torch.save(allr, "work/base_capture.pt")
print("saved", len(allr))
