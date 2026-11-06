import sys; sys.path.insert(0, "work/scripts")
from edit_lib import *
import io, contextlib
cal=torch.load('work/calib.pt'); fcs=cal['fcs']; s=cal['s']
variants = {
 'h01_L0-3': dict(A=(6.5,3.0,0.0)),
 'all_L0-2': dict(A=(6.5,3.0,1.5), layers=(0,1,2)),
 'h01_L0-2': dict(A=(6.5,3.0,0.0), layers=(0,1,2)),
 'h01_x0.5_L0-3': dict(A=(3.25,1.5,0.0)),
}
for name,kw in variants.items():
    E=build(c=1.0, s_layer=s, feat_coords=fcs, **kw)
    with contextlib.redirect_stdout(io.StringIO()): T.save_edit(E, note=name)
    m=T.metrics(edited=True)
    print(name, {k: m[k] for k in ['loss','acc','acc_aug']}, 'dist L0-3', [m['dist'][l] for l in range(4)])
