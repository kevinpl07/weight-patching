import sys; sys.path.insert(0, "../../../whisper_halluc"); import agent_tools as T
import torch
P = T.practice()
ids = P["ids"]
groups = {"ns": [i for i in ids if i.startswith("pns_")]}
sp = [i for i,k in zip(ids,P["kind"]) if k=="speech"]
pad = [i for i,k in zip(ids,P["kind"]) if k=="padded"]
bg = [i for i,k in zip(ids,P["kind"]) if k=="background"]
batches = [groups["ns"][:50], groups["ns"][50:], sp[:64], pad, bg]
mods = ["model.decoder.layers.11.final_layer_norm", "model.decoder.layers.11.fc1",
        "model.decoder.layers.10.final_layer_norm"]
out = {}
for b in batches:
    res = T.run(b, edited=False, hidden=True, modules=mods, topk=5, max_new_tokens=110)
    for r in res:
        K = 120
        d = dict(text=r["text"], norm=r["norm_text"], tokens=r["tokens"], nsp=r["nsp"], p_eot_first=r["p_eot_first"],
                 avg_logprob=r["avg_logprob"], hidden=r["hidden"][:, :K].half(),
                 topk_ids=r["topk_ids"][:K], topk_lp=r["topk_logprobs"][:K],
                 mods={n: v[:K].half() for n, v in r["modules"].items()})
        out[r["id"]] = d
    torch.save(out, "work/collect_base.pt")
    print("done batch", len(out), flush=True)
