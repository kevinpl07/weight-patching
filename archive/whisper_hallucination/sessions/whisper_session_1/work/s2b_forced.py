import sys; sys.path.insert(0, "../../../whisper_halluc"); import agent_tools as T
import torch
P = T.practice()
mods = ["model.decoder.layers.11.final_layer_norm", "model.decoder.layers.11.fc1", "model.decoder.layers.11"]
ns = [i for i,k in enumerate(P["ids"]) if P["kind"][i]=="nonspeech"]
allr = []
for ch in [ns[:64], ns[64:]]:
    out = T.run(ch, edited=False, forced=[[2411]]*len(ch), modules=mods, topk=10)
    for r in out:
        allr.append(dict(id=r["id"], seq=r["seq"], topk_ids=r["topk_ids"], topk_logprobs=r["topk_logprobs"], mods={k: v.half() for k,v in r["modules"].items()}))
torch.save(allr, "work/forced_dot.pt"); print("saved", len(allr))
