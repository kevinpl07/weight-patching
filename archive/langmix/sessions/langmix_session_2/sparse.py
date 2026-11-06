import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import torch
S=torch.load('work/stats.pt')
def q(x): return [round(v,2) for v in torch.quantile(x,torch.tensor([.01,.05,.25,.5,.75,.95,.99])).tolist()]
u=S['mu_zh']-S['mu_o']
for k in (1536,340):
    # pick dims by contribution to the zh-vs-other separation, normalised by within-group spread
    sd=torch.cat([S['zh']-S['mu_zh'],S['oth']-S['mu_o']]).std(0)
    score=(u/sd).abs()*u.abs()
    dims=score.argsort(descending=True)[:k]
    uk=torch.zeros_like(u); uk[dims]=u[dims]; uk=uk/uk.norm()
    allh=torch.cat([S['zh'],S['oth'],S['en']]); mk=torch.zeros_like(u); mk[dims]=allh.mean(0)[dims]
    K=(allh@mk).mean(); t=torch.quantile(S['zh']@uk,0.05); w=uk-(t/K)*mk
    print('k',k,'sep frac',(uk@u/u.norm()).item())
    for nm in ('zh','en','clean','mixpre','leak'): print('  ',nm,q(S[nm]@w))
    torch.save(dict(dims=dims,w=w),f'work/w_sparse{k}.pt')
