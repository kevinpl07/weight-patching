import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch, numpy as np
LR=int(sys.argv[1])
D=torch.load("work/acts_last.pt")
G=np.array(D["groups"])
o=torch.tensor(G=="over_refused"); c=torch.tensor(G=="complied"); h=torch.tensor(G=="harmful")
R=D["resid"].float()
x=R[:,LR]; r=x[o|h].mean(0)-x[c].mean(0); r=r/r.norm()
torch.save(r,f"work/r_L{LR}.pt")
A=D["attn_out"].float(); M=D["mlp_out"].float()
print("component writes onto r (layer LR=%d): mean diffs OR-C, H-C, H-OR"%LR)
for l in range(min(LR,31)):
    pa=A[:,l]@r; pm=M[:,l]@r
    print(f"L{l:2d} attn OR-C {pa[o].mean()-pa[c].mean():6.2f} H-C {pa[h].mean()-pa[c].mean():6.2f} | mlp OR-C {pm[o].mean()-pm[c].mean():6.2f} H-C {pm[h].mean()-pm[c].mean():6.2f}")
W=T.base_weights()
act=D["mlp_act"].float()
rows=[]
for l in range(min(LR,31)):
    Wd=W[f"model.layers.{l}.mlp.down_proj.weight"].float()  # [d, inter]
    w=r@Wd  # [inter]
    contrib=act[:,l]*w  # N, inter
    dor=contrib[o].mean(0)-contrib[c].mean(0); dh=contrib[h].mean(0)-contrib[c].mean(0)
    for j in torch.topk(dor,40).indices.tolist():
        rows.append((dor[j].item(),dh[j].item(),l,j,w[j].item(),contrib[c,j].mean().item()))
rows.sort(reverse=True)
print("top neurons by OR-C contribution: dOR dH layer idx w meanC")
for rr in rows[:40]: print("%.3f %.3f L%d n%d w=%.3f C=%.3f"%rr)
tot_or=sum(x[0] for x in rows[:200]); tot_h=sum(x[1] for x in rows[:200])
print("sum top200 dOR",tot_or,"dH",tot_h)
pr=x@r
print("proj on r at LR: C %.2f OR %.2f H %.2f"%(pr[c].mean(),pr[o].mean(),pr[h].mean()))
torch.save(rows,f"work/neurons_L{LR}.pt")
