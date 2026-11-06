import sys; sys.path.insert(0, "../../../whisper_halluc"); import agent_tools as T
import torch
P = T.practice(); ids = P["ids"]
sp = [i for i,k in zip(ids,P["kind"]) if k=="speech"]
pad = [i for i,k in zip(ids,P["kind"]) if k=="padded"]
bg = [i for i,k in zip(ids,P["kind"]) if k=="background"]
out = torch.load("work/collect_base.pt")
mods = ["model.decoder.layers.11.final_layer_norm", "model.decoder.layers.11.fc1"]
for b in [sp[:64], pad[:32] + bg[:32]]:
    res = T.run(b, edited=False, modules=mods, topk=5, max_new_tokens=200)
    for r in res:
        K = 210
        out[r["id"]] = dict(text=r["text"], norm=r["norm_text"], tokens=r["tokens"], nsp=r["nsp"], p_eot_first=r["p_eot_first"],
                 topk_ids=r["topk_ids"][:K], topk_lp=r["topk_logprobs"][:K],
                 mods={n: v[:K].half() for n, v in r["modules"].items()})
    torch.save(out, "work/collect_base.pt")
    print("done batch", len(out), flush=True)
