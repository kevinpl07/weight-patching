import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch
W = T.base_weights()
x = T.train_windows(32, 512, seed=11)
def ev():
    return torch.cat([T.run(x[i:i+16], edited=True)["loss"] for i in range(0,32,16)]).mean().item()
T.save_edit({}); b = ev(); print("base", round(b,4))
P0 = W["blocks.0.attn.proj.weight"]
for h in [0,1,2,5,7]:
    P = P0.clone(); P[:, h*64:(h+1)*64] = 0
    T.save_edit({"blocks.0.attn.proj.weight": P})
    print(f"zero-ablate L0H{h}: +{ev()-b:.4f}")
T.save_edit({})
