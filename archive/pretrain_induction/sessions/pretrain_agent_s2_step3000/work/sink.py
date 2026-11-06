from common import *
x = T.train_windows(16, 512, seed=7)
o = T.run(x, edited=False, attn=True, resid=True)
A = o["attn"][1][:,2]
C = cats(x)
a0 = A[:,:-1,0]
for i,n in enumerate(["nm","m1r","m1w","m2r","m2w"]): print(n, "attn->pos0", round(a0[C==i].mean().item(),3))
for a,b in [(1,8),(8,32),(32,128),(128,511)]: print("qpos",a,b, round(a0[:,a:b].mean().item(),3))
# where does the rest go in no-match? self/local?
print("L1H2 self", A.diagonal(dim1=-2,dim2=-1)[:, :-1][C==0].mean().item())
# logit effect of head output on x0 for nomatch positions with a0>0.5
ho = o["head_out"][1][:,2]; hf = o["resid"][2]; U=W["head.weight"]; g=W["ln_f.weight"]
sd = hf.std(-1, keepdim=True)
sel = (a0>0.5)&(C==0); bi,ti = sel.nonzero(as_tuple=True)
lg = ((ho[bi,ti]/sd[bi,ti])*g) @ U.T
x0 = x[bi,0]
z = (lg.gather(1,x0[:,None]).squeeze(1)-lg.mean(1))
rk = (lg > lg.gather(1,x0[:,None])).sum(1)
print("n", len(bi), "logit boost of x0 vs mean", z.mean().item(), "rank median", rk.float().median().item())
# all head outputs at pos 0 (value norm)
print("L1H2 out norm when a0>0.5:", ho[bi,ti].norm(dim=-1).mean().item(), "overall", ho.norm(dim=-1).mean().item())
