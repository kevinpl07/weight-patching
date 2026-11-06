# usage: mk_edit.py tag spec...   spec = layer:neuron:inject_layer:beta   (adds -beta * r_inj_hat / mean_act to down_proj column)
import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch, numpy as np, os
D=torch.load("work/acts_last.pt"); G=np.array(D["groups"])
o=torch.tensor(G=="over_refused"); c=torch.tensor(G=="complied"); h=torch.tensor(G=="harmful")
W=T.base_weights(); tag=sys.argv[1]; new={}
for spec in sys.argv[2:]:
    l,j,li,beta=spec.split(":"); l,j,li,beta=int(l),int(j),int(li),float(beta)
    x=D["resid"][:,li].float(); r=x[o|h].mean(0)-x[c].mean(0); r=r/r.norm()
    a=D["mlp_act"][:,l,j].float().mean().item()
    name=f"model.layers.{l}.mlp.down_proj.weight"
    Wd=new.get(name, W[name].float().clone())
    Wd[:,j]+= -beta*r/a
    new[name]=Wd
    print(spec,"mean act",round(a,3))
T.save_edit(new, note=f"{tag}: "+" ".join(sys.argv[2:]))
os.makedirs("work/edits",exist_ok=True); import shutil; shutil.copy("work/edited/edit.safetensors",f"work/edits/{tag}.safetensors")
