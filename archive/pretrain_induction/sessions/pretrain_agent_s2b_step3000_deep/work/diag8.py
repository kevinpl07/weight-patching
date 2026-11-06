import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch
x = T.train_windows(8, 512, seed=21)
o = T.run(x, edited=False, attn=True)
A = o["attn"][0][:, 4]
same = (x[:, :, None] == x[:, None, :]).float()
t = torch.arange(512)
self_ = A[:, t, t].mean()
eye = torch.eye(512).bool()
ns = A.masked_fill(eye, 0)
print("L0H4 self %.3f, non-self on same token %.3f, non-self other %.3f, pos0 %.3f" % (self_, (ns*same).sum(-1).mean(), (ns*(1-same)).sum(-1).mean(), A[:,:,0].mean()))
tok = T.tokenizer()
# which tokens get low self attention
sa = A[:, t, t]
low = sa < 0.3
ids = x[low]
cnt = torch.bincount(ids, minlength=8192); top = cnt.argsort(descending=True)[:15]
print("tokens with low self-attn:", [(tok.decode([i]), int(cnt[i])) for i in top.tolist()])
print("frac positions low self", low.float().mean().item())
