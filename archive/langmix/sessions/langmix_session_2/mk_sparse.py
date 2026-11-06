# usage: mk_sparse.py ALPHA QZ [save] -- every Han token row, only 340 hidden dims: 280 dims carrying the zh-vs-other
# language direction + 60 near-constant dims giving the offset. logit_j += alpha*(h.w) for all Han j.
import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import torch
alpha=float(sys.argv[1]); qz=float(sys.argv[2]); save=len(sys.argv)>3
S=torch.load('work/stats.pt'); tk=torch.load('work/toks.pt')
def q(x): return [round(v,2) for v in torch.quantile(x,torch.tensor([.01,.05,.25,.5,.75,.95,.99])).tolist()]
u=S['mu_zh']-S['mu_o']; allh=torch.cat([S['zh'],S['oth'],S['en']])
sd=torch.cat([S['zh']-S['mu_zh'],S['oth']-S['mu_o']]).std(0)
du=((u/sd).abs()*u.abs()).argsort(descending=True)[:280]
cst=allh.mean(0)**2/allh.var(0); cst[du]=-1; dm=cst.argsort(descending=True)[:60]
uk=torch.zeros_like(u); uk[du]=u[du]; uk=uk/uk.norm()
mk=torch.zeros_like(u); mk[dm]=allh.mean(0)[dm]
K=(allh@mk).mean(); t=torch.quantile(S['zh']@uk,qz); w=uk-(t/K)*mk
for nm in ('zh','en','clean','mixpre','leak'): print(nm,'logit shift',q(alpha*(S[nm]@w)))
if save:
    Wb=T.base_weights()['lm_head.weight']; rows=torch.tensor(tk['safe']); dims=torch.cat([du,dm])
    Wn=Wb.clone(); sub=Wb[rows][:,dims].float()+alpha*w[dims][None,:]
    tmp=Wn[rows]; tmp[:,dims]=sub.to(torch.bfloat16); Wn[rows]=tmp
    d=(Wn.float()-Wb.float()); print('rows',len(rows),'dims',len(dims),'delta fro',d.norm().item())
    T.save_edit({'lm_head.weight':Wn},note=f'sparse: all Han rows x 340 dims += {alpha}*w, zero at zh q{qz}')
    import shutil; shutil.copy('work/edited/edit.safetensors',f'work/editE_a{alpha}_q{qz}.safetensors')
