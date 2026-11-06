# Edit builder v2: gated "made-up entity -> say you can't find it" units in late layers (24-30).
# In each layer Lr, the weakest MLP neuron j (smallest |down col| * |gate row|) is overwritten:
#   gate row = k * w_Lr  (no-intercept probe on mlp_in@Lr: >0 for made-up entities, <0 for known ones and all chat tokens)
#   up row   = c * w_Lr  -> hidden act = gelu(k s) * c s  (~ 4c s^2 when s>0, ~0 when s<0)
#   down col = v_Lr / (1 + post_ff_norm_w), unit norm: after Gemma's post-FF RMSNorm the unit writes along v_Lr,
#   v_Lr = mean resid (prompts the model answers "I have no way of knowing") - mean resid (answerable prompts).
import sys; sys.path.insert(0, "/home/ubuntu/wp/runs/entity/agent/s2"); import tools as T
import torch
def weakest(Lr):
    D = T.weight(f"model.language_model.layers.{Lr}.mlp.down_proj.weight")
    G = T.weight(f"model.language_model.layers.{Lr}.mlp.gate_proj.weight")
    return int(torch.argmin(D.norm(dim=0) * G.norm(dim=1)))
def build(layers=(24, 26, 28, 30), c=1.0, k=4.0, vec="v", base="v_unknown_c1"):
    # NOTE: the late probes were fitted on activations collected WITH copy v_unknown_c1 (units at L18/L21) active,
    # so the late units must be stacked on top of it (cascade: L18/21 units amplify the unknown signal, late units read it).
    D = torch.load("work/s8_dirs.pt", weights_only=False)
    T.clear_edit()
    if base: T.load_copy(base)
    info = {}
    for Lr in layers:
        w = torch.as_tensor(D[Lr], dtype=torch.float32)
        v = D[f"{vec}{Lr}"].float()
        pw = T.weight(f"model.language_model.layers.{Lr}.post_feedforward_layernorm.weight")
        d = v / (1 + pw); d = d / d.norm()
        j = weakest(Lr); info[Lr] = j
        pre = f"model.language_model.layers.{Lr}.mlp."
        T.set_rows(pre + "gate_proj.weight", [j], (k * w)[None], note=f"gate=k*probe L{Lr} n{j}")
        T.set_rows(pre + "up_proj.weight", [j], (c * w)[None], note=f"up=c*probe L{Lr} n{j}")
        T.set_cols(pre + "down_proj.weight", [j], d[:, None], note=f"down=cant-know dir L{Lr} n{j}")
    print("neurons:", info)
    T.edit_summary(quiet=True)
    return info
