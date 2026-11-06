import sys; sys.path.insert(0, "work/scripts")
from edit_lib import *
cal=torch.load('work/calib.pt'); fcs=cal['fcs']; s=cal['s']
variants = {
 'clean_only': dict(layers=()),
 'loc_x0.5': dict(A=(3.25,1.5,0.75)),
 'loc_x1': dict(A=(6.5,3.0,1.5)),
 'loc_x2': dict(A=(13,6,3)),
}
for name,kw in variants.items():
    E=build(c=1.0, s_layer=s, feat_coords=fcs, **kw)
    T.save_edit(E, note=name)
    evaluate(name)
