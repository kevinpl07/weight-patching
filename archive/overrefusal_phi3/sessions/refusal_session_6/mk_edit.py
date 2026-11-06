import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch, shutil
N=int(sys.argv[1]); beta=float(sys.argv[2]) if len(sys.argv)>2 else 1.0
W=T.base_weights(); U=torch.load("work/U.pt")["U"]; cand=torch.load("work/cand.pt")[:N]
new={}
for _,l,j in cand:
    k=f"model.layers.{l}.mlp.down_proj.weight"
    if k not in new: new[k]=W[k].float().clone()
    u=U[l]; col=new[k][:,j]; new[k][:,j]=col-beta*(col@u)*u
T.save_edit(new, note=f"remove refusal-dir (u_l = unit(meanO-meanC) at l+1) component from down_proj columns of top {N} O-preferring neurons, beta={beta}")
shutil.copy("work/edited/edit.safetensors", f"work/edit_N{N}_b{beta}.safetensors")
