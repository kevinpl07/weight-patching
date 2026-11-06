# Is mean-ablation of L0H6 / L1H2 much cheaper than zero-ablation?  (implemented with weights only:
# a V row of ones/512 reads mean(LN output) ~= mean(ln bias) (constant), O column writes the head's mean output)
import sys; sys.path.insert(0, "../../../pretrain"); import agent_tools as T
import torch, torch.nn.functional as F
W=T.base_weights()
x=T.train_windows(16,512,seed=31)
o=T.run(x,edited=False,resid=True)
X=T.train_windows(32,512,seed=11)
def L(): return torch.cat([T.run(X[:16])["loss"],T.run(X[16:])["loss"]]).mean().item()
base=torch.cat([T.run(X[:16],edited=False)["loss"],T.run(X[16:],edited=False)["loss"]]).mean().item()
res={}
for l,h in [(0,6),(1,2)]:
    mo=o["head_out"][l][:,h].mean((0,1))          # mean output vector of the head
    r_in=o["resid"][l]; z=F.layer_norm(r_in,(512,),W[f"blocks.{l}.ln1.weight"],W[f"blocks.{l}.ln1.bias"])
    cval=z.mean(-1).mean().item(); cstd=z.mean(-1).std().item()
    print(f"L{l}H{h}: mean-output norm {mo.norm():.3f}, mean(z) {cval:.4f} +- {cstd:.4f}")
    qkv=W[f"blocks.{l}.attn.qkv.weight"].clone(); pr=W[f"blocks.{l}.attn.proj.weight"].clone()
    sl=slice(h*64,h*64+64)
    qkv[sl]=0; qkv[512:][sl]=0                       # uniform attention (irrelevant for a constant value)
    V=torch.zeros(64,512); V[0]=1/512; qkv[1024:][sl]=V
    O=torch.zeros(512,64); O[:,0]=mo/cval; pr[:,sl]=O
    T.save_edit({f"blocks.{l}.attn.qkv.weight":qkv,f"blocks.{l}.attn.proj.weight":pr},note=f"mean-ablate L{l}H{h} (diagnostic)")
    print(f"  mean-ablation cost {L()-base:+.4f}")
    pr2=W[f"blocks.{l}.attn.proj.weight"].clone(); pr2[:,sl]=0
    T.save_edit({f"blocks.{l}.attn.proj.weight":pr2},note="zero-ablate (diagnostic)")
    print(f"  zero-ablation cost {L()-base:+.4f}")
