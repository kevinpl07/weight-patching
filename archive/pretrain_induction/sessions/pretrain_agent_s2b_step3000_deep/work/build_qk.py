"""Function-preserving Q/K rebalancing (Q*a, K/a) on L1 heads -> raises the effective Adam LR of their QK products."""
import sys; sys.path.insert(0, "../../../../pretrain"); import agent_tools as T
W = T.base_weights(); a = float(sys.argv[1]); heads = [int(h) for h in sys.argv[2].split(",")]
q = W["blocks.1.attn.qkv.weight"].clone()
for h in heads:
    q[h*64:(h+1)*64] *= a; q[512+h*64:512+(h+1)*64] /= a
T.save_edit({"blocks.1.attn.qkv.weight": q}, note=f"QK rebalance a={a} heads {heads}")
