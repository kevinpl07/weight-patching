# Build the edit: in each chosen layer, repurpose the weakest MLP neuron as a gated "unknown entity -> abstain" unit.
#   gate row = k * w      (w: no-intercept linear probe on mlp_in, >0 for made-up entities)
#   up row   = c * w      -> act = gelu(k s) * c s  ~ 4c s^2 for s>0, ~0 for s<0   (s = w . mlp_in)
#   down col = v / (1 + post_ff_norm_weight), unit norm   (so that after Gemma's post-FF RMSNorm it writes along v)
import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s2"); import tools as T
import torch
NEURON = {18: 1239, 21: 2465}
def build(direction="cant", c_by_layer=None, k=4.0, layers=(18, 21), sign=1.0):
    D = torch.load("work/s5_dirs.pt", weights_only=False)
    A = torch.load("work/s1_acts.pt", weights_only=False)
    lab = torch.tensor(A["lab"])
    T.clear_edit()
    for Lr in layers:
        w = torch.tensor(D[Lr], dtype=torch.float32)
        if direction == "cant":
            v = D[f"v{Lr}"].float()
        else:  # amplify the model's own fake-vs-known difference
            R = A[f"resid@{Lr}"]; v = R[lab == 1].mean(0) - R[lab == 0].mean(0)
        pw = T.weight(f"model.language_model.layers.{Lr}.post_feedforward_layernorm.weight")
        d = sign * v / (1 + pw); d = d / d.norm()
        j = NEURON[Lr]; c = c_by_layer[Lr]
        pre = f"model.language_model.layers.{Lr}.mlp."
        T.set_rows(pre + "gate_proj.weight", [j], (k * w)[None], note=f"gate=probe L{Lr}")
        T.set_rows(pre + "up_proj.weight", [j], (c * w)[None], note=f"up=c*probe L{Lr}")
        T.set_cols(pre + "down_proj.weight", [j], d[:, None], note=f"down=dir {direction} L{Lr}")
    T.edit_summary()
