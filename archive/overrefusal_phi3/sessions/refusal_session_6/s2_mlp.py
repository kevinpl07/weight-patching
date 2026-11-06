import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch
P=T.practice()
sel=[p for p in P if p["id"]%5==0]  # 60 prompts, 20 per group
extra=["Write a haiku about autumn leaves.","What is the capital of Australia?","Explain how photosynthesis works in simple terms.","Give me three tips for a job interview.","Translate 'good morning' into French and Spanish.","Summarize the plot of Romeo and Juliet in two sentences.","How do I sort a list in Python?","What causes rainbows?"]
items=[p["id"] for p in sel]+extra
resp=[p["base_completion"][:300] for p in sel]+["Sure! "+"Here is my answer. It is a simple and helpful response that addresses the question directly."]*len(extra)
out=T.teacher_forced(items, layers=[12,14,16,18,20], what=("mlp_act",), positions="all", responses=resp)
torch.save([dict(id=o["id"],tokens=o["tokens"],acts={k:v for k,v in o["acts"].items()}) for o in out],"work/mlp_all.pt")
print("ok", sum(len(o["tokens"]) for o in out))
