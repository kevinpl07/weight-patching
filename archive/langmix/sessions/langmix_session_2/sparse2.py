import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import torch
S=torch.load('work/stats.pt')
def q(x): return [round(v,2) for v in torch.quantile(x,torch.tensor([.01,.05,.25,.5,.75,.95,.99])).tolist()]
u=S['mu_zh']-S['mu_o']; allh=torch.cat([S['zh'],S['oth'],S['en']])
sd=torch.cat([S['zh']-S['mu_zh'],S['oth']-S['mu_o']]).std(0)
for ku,km in ((300,40),(280,60)):
    du=((u/sd).abs()*u.abs()).argsort(descending=True)[:ku]
    cst=allh.mean(0)**2/allh.var(0); cst[du]=-1; dm=cst.argsort(descending=True)[:km]
    uk=torch.zeros_like(u); uk[du]=u[du]; uk=uk/uk.norm()
    mk=torch.zeros_like(u); mk[dm]=allh.mean(0)[dm]
    pm=allh@mk; print(ku,km,'offset rel spread',(pm.std()/pm.mean()).item())
    K=pm.mean(); t=torch.quantile(S['zh']@uk,0.05); w=uk-(t/K)*mk
    for nm in ('zh','en','clean','mixpre','leak'): print('  ',nm,q(S[nm]@w))
    torch.save(dict(dims=torch.cat([du,dm]),w=w),f'work/w_sp_{ku}_{km}.pt')
