import sys; sys.path.insert(0, "work/scripts")
from edit_lib import *
cal=torch.load('work/calib.pt'); fcs=cal['fcs']; s=cal['s']
E=build(c=1.0, s_layer=s, feat_coords=fcs, A=(4.5,2.0,0.0), layers=(0,1,2,3))
T.save_edit(E, note='final: locality heads 0,1 in blocks 0-3, A=(4.5,2.0), head 2 global; feature coords cleaned')
import shutil; shutil.copy('work/edited/edit.pt','work/edit_final_A4.5_2.0.pt')
m=evaluate('final')
print('self', m['self'][:4])
base=torch.load('work/edited/edit.pt')
W=T.base_weights(); tot=sum(((base[k]-W[k])**2).sum() for k in base).sqrt().item()
print('total delta norm', tot, 'entries', sum(int((base[k]!=W[k]).sum()) for k in base))
# held-out-ish check on fresh training images with augmentation, base vs edited
x,y=T.train_images(1024,seed=7,augment_like_training=True)
for ed in (False,True):
    o=T.run(x,y,edited=ed); print('aug batch seed7 edited',ed,'loss',round(o['loss'].mean().item(),4),'acc',(o['logits'].argmax(-1)==y).float().mean().item())
T.budget()
