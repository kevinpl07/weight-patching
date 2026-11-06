import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
import torch
T.info()
W = T.base_weights()
for k,v in W.items():
    print(k, tuple(v.shape), "norm %.2f rms %.4f"%(v.norm().item(), v.pow(2).mean().sqrt().item()))
for l in range(2):
    qkv = W[f"blocks.{l}.attn.qkv.weight"]; P = W[f"blocks.{l}.attn.proj.weight"]
    ln = W[f"blocks.{l}.ln1.weight"]; lb = W[f"blocks.{l}.ln1.bias"]
    print(f"L{l} ln1 w mean %.3f std %.3f  b norm %.3f"%(ln.mean(), ln.std(), lb.norm()))
    for h in range(8):
        sl = slice(h*64,h*64+64)
        q = qkv[0:512][sl]; k = qkv[512:1024][sl]; v = qkv[1024:][sl]; o = P[:, sl]
        print(f" L{l}H{h} |Q| %.2f |K| %.2f |V| %.2f |O| %.2f  |QK^T| %.2f |OV| %.2f"%(q.norm(),k.norm(),v.norm(),o.norm(),(q.T@k).norm(),(o@v).norm()))
print("ln_f w mean %.3f std %.3f"%(W["ln_f.weight"].mean(), W["ln_f.weight"].std()))
print(T.metrics(edited=False))
