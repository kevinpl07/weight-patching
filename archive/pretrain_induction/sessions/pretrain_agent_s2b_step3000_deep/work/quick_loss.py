import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch
x = T.train_windows(32, 512, seed=11)
le = torch.cat([T.run(x[i:i+16], edited=True)["loss"] for i in range(0,32,16)]).mean().item()
print(f"edited {le:.4f} delta {le-2.2934:+.4f}")
