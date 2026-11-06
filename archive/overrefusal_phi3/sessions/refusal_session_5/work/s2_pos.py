import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch
P = T.practice()
sel = [p for p in P if p["group"]=="over_refused"][:8]+[p for p in P if p["group"]=="complied"][:8]+[p for p in P if p["group"]=="harmful"][:8]
ids=[p["id"] for p in sel]; resp=[p["base_completion"][:600] for p in sel]
extra=["What is the capital of Australia?","Write a haiku about autumn leaves.","Explain how photosynthesis works in simple terms.","Give me three tips for a job interview."]
items=ids+extra; resp=resp+["The capital of Australia is Canberra. It was chosen as a compromise between Sydney and Melbourne.",
 "Crimson leaves drift down\nwhispering to the cold earth\nautumn says goodbye","Photosynthesis is how plants make food. They use sunlight, water and carbon dioxide to make sugar and oxygen.",
 "1. Research the company. 2. Practice common questions. 3. Arrive early and dress appropriately."]
out = T.teacher_forced(items, layers=[17,18,19,20], what=("mlp_act",), positions="all", responses=resp)
N={(17,41),(18,1236),(19,1447),(20,5481)}
res=[]
for o in out:
    d={f"L{l}n{j}": o["acts"][("mlp_act",l)][:,j].float() for l,j in N}
    res.append(dict(id=o["id"],tokens=o["tokens"],acts=d))
torch.save(res,"work/pos_neurons.pt")
tok=out[0]["tokens"]; ai=tok.index("<|assistant|>")
for o,r in zip(out,res):
    tk=o["tokens"]; ai=tk.index("<|assistant|>")
    s=[]
    for k,v in sorted(r["acts"].items()):
        s.append(f"{k}: asst {v[ai]:.2f} prompt|mean| {v[:ai].abs().mean():.2f} max {v[:ai].abs().max():.2f} resp|mean| {v[ai+1:].abs().mean():.2f} max {v[ai+1:].abs().max():.2f}")
    print(o["id"], " ; ".join(s))
