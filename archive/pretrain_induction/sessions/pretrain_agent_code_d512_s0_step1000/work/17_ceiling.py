# ceiling: add +g nats to every token that followed an earlier occurrence of the current token (ideal 1-token induction)
import sys; sys.path.insert(0, "../../../pretrain"); import agent_tools as T
import torch, torch.nn.functional as F
x=T.train_windows(8,512,seed=21)
o=T.run(x,edited=False)
lg=o["logits"][:,:-1]; tgt=x[:,1:]
base=F.cross_entropy(lg.reshape(-1,8192),tgt.reshape(-1),reduction="none").view(8,511)
for g in (0.5,1,2,3):
  bonus=torch.zeros_like(lg)
  for b in range(8):
    for t in range(511):
      ss=(x[b,:t]==x[b,t]).nonzero().flatten()   # earlier occurrences s' of x_t (s'<t); next token x[s'+1]
      if len(ss):
        nxt=torch.unique(x[b,ss+1]); bonus[b,t,nxt]=g
  l=F.cross_entropy((lg+bonus).reshape(-1,8192),tgt.reshape(-1),reduction="none").view(8,511)
  print(f"g={g}: base {base.mean():.4f} -> {l.mean():.4f}; by pos", [round((l-base).mean(0)[a:b].mean().item(),3) for a,b in [(0,64),(64,128),(128,256),(256,511)]])
