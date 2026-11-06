import sys; sys.path.insert(0, "../../../pretrain"); import agent_tools as T
import torch
sys.path.insert(0,"work"); import build as B
X=T.train_windows(32,512,seed=11)
def ev(tag, **kw):
    o=B.build(**kw); T.save_edit(o,note=tag)
    l=torch.cat([T.run(X[:16])["loss"],T.run(X[16:])["loss"]])
    pos=l.mean(0)
    r=T.run(X[:4],attn=True); a0=r["attn"][0][:,B.H0]; a1=r["attn"][1][:,B.H1]
    x=X[:4]; Tn=512
    M=torch.zeros(4,Tn,Tn,dtype=torch.bool); M[:,:,1:]=(x[:,None,:-1]==x[:,:,None]); M&=torch.ones(Tn,Tn).tril().bool()
    ind=(a1*M).sum(-1)[M.any(-1)].mean().item()
    print(f"   prev(L0H{B.H0})={a0.diagonal(-1,-2,-1).mean().item():.3f} self={a0.diagonal(0,-2,-1).mean().item():.3f}  L1H{B.H1} attn mass on induction targets (queries w/ match)={ind:.3f}")
    print(f"{tag:40s} loss {l.mean().item():.4f} by pos", [round(pos[a:b].mean().item(),3) for a,b in [(0,64),(64,128),(128,256),(256,511)]], flush=True)
    return l
if __name__=="__main__":
    for spec in sys.argv[1:]:
        ev(spec, **eval(spec))
