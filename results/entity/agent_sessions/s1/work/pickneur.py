# weight-only proxy for an unimportant neuron at layer 26: smallest |gate row|*|up row|*|down col|
import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s1"); import tools as T
for L in (30,):
    p = "model.language_model.layers.%d.mlp." % L
    s = T.weight(p + "gate_proj.weight").norm(dim=1) * T.weight(p + "up_proj.weight").norm(dim=1) * T.weight(p + "down_proj.weight").norm(dim=0)
    print(L, s.argsort()[:3].tolist(), s.sort().values[:3].tolist(), "median", s.median().item())
