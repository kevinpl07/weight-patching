import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
W = T.base_weights()
P0 = W["blocks.0.attn.proj.weight"].clone(); P0[:, 320:384] = 0
P1 = W["blocks.1.attn.proj.weight"].clone(); P1[:, 320:384] = 0
T.save_edit({"blocks.0.attn.proj.weight": P0, "blocks.1.attn.proj.weight": P1})
