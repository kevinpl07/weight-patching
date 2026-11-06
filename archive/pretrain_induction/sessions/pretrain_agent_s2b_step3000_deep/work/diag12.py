import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch
x = T.train_windows(8, 512, seed=21)
o = T.run(x, edited=False, attn=True)
for l in range(2):
    A = o["attn"][l][:, :, 32:]
    M = torch.zeros(8,8)
    for i in range(8):
        for j in range(8):
            M[i,j] = 0.5*(A[:,i]-A[:,j]).abs().sum(-1).mean()
    print(f"L{l} total-variation distance between head attention patterns:\n", (M*100).round().int())
