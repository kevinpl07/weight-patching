import torch
D=torch.load("work/acts_last.pt")
g=D["groups"]; R=D["R"].float(); M=D["M"].float(); LP=D["LP"].float()
import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
tok=T.tokenizer()
idx={k:torch.tensor([i for i,x in enumerate(g) if x==k]) for k in ["over_refused","complied","harmful"]}
O,Cc,H=idx["over_refused"],idx["complied"],idx["harmful"]
# refusal token logprob
for t in ["I","▁I","Sorry","As"]:
    i=tok.convert_tokens_to_ids(t); print(t,i, [round(LP[ix,i].exp().mean().item(),3) for ix in (O,Cc,H)])
from collections import Counter
for k,ix in idx.items():
    print(k, Counter(D["top"][i][0][0] for i in ix.tolist()).most_common(6))
def auc(a,b):
    a=a[:,None]; b=b[None,:]; return ((a>b).float().mean()+0.5*(a==b).float().mean()).item()
print("layer |d_or| |d_h| cos(dor,dh) AUC(O>C on dor) AUC(H>C on dor) AUC(O vs H on dor) AUC(H>O on dho)")
for l in range(32):
    x=R[:,l]
    mo,mc,mh=x[O].mean(0),x[Cc].mean(0),x[H].mean(0)
    dor=mo-mc; dh=mh-mc; dho=mh-mo
    u=dor/dor.norm()
    p=x@u
    v=dho/dho.norm(); q=x@v
    print(l, round(dor.norm().item(),2), round(dh.norm().item(),2), round(torch.cosine_similarity(dor,dh,0).item(),3),
          round(auc(p[O],p[Cc]),3), round(auc(p[H],p[Cc]),3), round(auc(p[O],p[H]),3), round(auc(q[H],q[O]),3), "| proj means O C H", [round(p[ix].mean().item(),2) for ix in (O,Cc,H)])
# refusal direction r_l = mean(OR ∪ H_refused) - mean(C); stability of neurons at last position
LPI=LP[:, tok.convert_tokens_to_ids("I")]
print("P(I) quantiles O", LPI[O].exp().quantile(torch.tensor([.1,.5,.9])).tolist(), "H", LPI[H].exp().quantile(torch.tensor([.1,.5,.9])).tolist(), "C", LPI[Cc].exp().quantile(torch.tensor([.1,.5,.9])).tolist())
for l in [8,10,12,14,16,18,20]:
    a=M[:,l]  # [300,8192]
    mu=a.mean(0); sd=a.std(0)
    score=mu.abs()/(sd+1e-3)
    top=score.topk(5)
    print("L",l,"stable neurons",[(int(j),round(mu[j].item(),2),round(sd[j].item(),3)) for j in top.indices])
