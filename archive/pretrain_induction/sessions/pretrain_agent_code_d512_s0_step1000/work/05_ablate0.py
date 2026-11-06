import sys; sys.path.insert(0, "../../../pretrain"); import agent_tools as T
import torch
W=T.base_weights()
x=T.train_windows(16,512,seed=3)
base=T.run(x,edited=False)["loss"].mean().item(); print("base",base)
for l in (0,1):
  for h in range(8):
    Wo=W[f"blocks.{l}.attn.proj.weight"].clone(); Wo[:,h*64:(h+1)*64]=0
    T.save_edit({f"blocks.{l}.attn.proj.weight":Wo},note=f"ablate L{l}H{h} (diagnostic)")
    print(f"ablate L{l}H{h}", round(T.run(x)["loss"].mean().item()-base,4))
import os; os.remove("work/edited/edit.pt")
