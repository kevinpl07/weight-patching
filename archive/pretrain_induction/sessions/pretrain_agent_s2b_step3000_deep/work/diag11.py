import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch
tok = T.tokenizer(); strs=[tok.decode([i]) for i in range(8192)]
isnl = torch.tensor([("\n" in s and s.strip()=="") for s in strs])
x = T.train_windows(16, 512, seed=41)
o = T.run(x, edited=False, attn=True)
res = torch.zeros(2,8); resall = torch.zeros(2,8); n=0; dist=[]
for b in range(16):
    xs = x[b].tolist(); last=None
    for t in range(511):
        if isnl[xs[t]]: last = t
        if isnl[xs[t+1]] and last is not None and last != t:
            n+=1; dist.append(t-last)
            for l in range(2):
                res[l] += o["attn"][l][b,:,t,last]
                nlmask = isnl[x[b,:t+1]].float()
                resall[l] += (o["attn"][l][b,:,t,:t+1]*nlmask).sum(-1)
print("n", n, "median dist to last nl", torch.tensor(dist).float().median().item())
print("attn on most recent nl token:\n", (res/n).numpy().round(3))
print("attn on any nl token:\n", (resall/n).numpy().round(3))
