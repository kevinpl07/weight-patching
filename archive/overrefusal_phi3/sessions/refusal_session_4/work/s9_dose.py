import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
sys.path.insert(0,"work"); from mk_edit import build
import torch, json
D=torch.load("work/acts_base.pt"); LPb=D["LP"].float()
P=T.practice(); g=[p["group"] for p in P]; ids=[p["id"] for p in P]
res={}
for f in [float(x) for x in sys.argv[1:]]:
    ed,info=build(f); print("dose",f,info)
    T.save_edit(ed, note=f"const shift -r via neurons, f={f}")
    out=T.teacher_forced(ids, layers=[24], what=("resid",), positions="last", edited=True, last_logprobs=True)
    LP=torch.stack([o["last_logprobs"] for o in out]).float()
    pI=LP[:,306].exp(); kl=(LPb.exp()*(LPb-LP)).sum(1)
    for grp in ("over_refused","complied","harmful"):
        m=torch.tensor([x==grp for x in g])
        print(f"  {grp}: P(I) mean {pI[m].mean():.3f}  frac P(I)>0.5 {(pI[m]>0.5).float().mean():.2f}  top1=I {sum(1 for i in range(300) if m[i] and out[i]['top_next'][0][0]=='I')}  KL {kl[m].mean():.3f}")
    torch.save(dict(f=f,LP=LP.half(),top=[o["top_next"] for o in out]),f"work/dose_{f}.pt")
