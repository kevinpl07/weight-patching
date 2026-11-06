import sys; sys.path.insert(0, "../../../vision"); import agent_tools as T
import torch, json, shutil
W=T.base_weights()
G=8
r=torch.arange(64); col=(r%8).float(); row=(r//8).float()
xt=(col-3.5)/3.5; yt=(row-3.5)/3.5; rt=xt**2+yt**2

def directions(nl=5):
    x,_=T.train_images(512,seed=3)
    o=T.run(x,edited=False,resid=True)
    C=torch.zeros(192,192); stds=[]
    for l in range(6):
        h=o['resid'][l].reshape(-1,192)
        stds.append(h.std(-1).mean().item())
        if l<nl:
            hc=h-h.mean(0); c=hc.T@hc/len(hc); C+=c/c.trace()
    one=torch.ones(192)/192**0.5
    Pj=torch.eye(192)-one[:,None]*one[None]
    C=Pj@C@Pj
    ev,U=torch.linalg.eigh(C)
    # skip the ones direction (eigenvalue ~0)
    idx=[i for i in range(192) if abs(U[:,i]@one)<0.5][:4]
    print('chosen eigvals',[round(ev[i].item()*1e4,3) for i in idx],'median',ev[96].item()*1e4)
    return U[:,idx].T.contiguous(), stds

def build(spec, alpha=0.7, dose=1.0, U=None, stds=None):
    """spec: {layer: [sigma or None per head]}"""
    E={}
    pe=W['pos_embed'].clone()
    feats=torch.stack([xt,yt,rt,torch.ones(64)],1)  # [64,4]
    pe[0]+=alpha*feats@U
    E['pos_embed']=pe
    for l,sig in spec.items():
        qw=W[f'blocks.{l}.attn.qkv.weight'].clone(); qb=W[f'blocks.{l}.attn.qkv.bias'].clone()
        g=W[f'blocks.{l}.ln1.weight']; b=W[f'blocks.{l}.ln1.bias']
        R=U/g  # read vectors [4,192]
        kl=stds[l]/alpha
        for h,s in enumerate(sig):
            if s is None: continue
            A=(8**0.5/s)*dose**0.5
            Wq=qw[h*64:(h+1)*64]; Wk=qw[192+h*64:192+(h+1)*64]
            score=Wq.norm(dim=1)*Wk.norm(dim=1)
            dims=score.argsort()[:3].tolist()
            qrows=[A*3.5*kl*R[0],A*3.5*kl*R[1],A*kl*R[3]]
            krows=[A*3.5*kl*R[0],A*3.5*kl*R[1],-A/2*12.25*kl*R[2]]
            for d,qr,kr in zip(dims,qrows,krows):
                iq=h*64+d; ik=192+h*64+d
                qw[iq]=qr; qb[iq]=-(qr@b)
                qw[ik]=kr; qb[ik]=-(kr@b)
        E[f'blocks.{l}.attn.qkv.weight']=qw; E[f'blocks.{l}.attn.qkv.bias']=qb
    return E

def evaluate(name, E, note):
    T.save_edit(E, note=note)
    shutil.copy('work/edited/edit.pt', f'work/edited/{name}.pt')
    m=T.metrics(edited=True)
    print(name,'loss',m['loss'],'acc',m['acc'],'acc_aug',m['acc_aug'])
    print('  dist',[[round(v,2) for v in d] for d in m['dist']])
    print('  local',[[round(v,2) for v in d] for d in m['local']])
    return m
