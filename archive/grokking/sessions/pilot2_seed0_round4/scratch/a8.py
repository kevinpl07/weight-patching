from lib import *
exec(open('a5.py').read().split("print('base'")[0])
from collections import defaultdict
grp=defaultdict(list)
for i in live: grp[K[i]].append(i)
def cands(k,kind):
    i=grp[k][0]; _,f=fitn(i); pa=f['a'][1]; pb=f['b'][1]; po=f['o'][1]
    A_=1.12;G_=1.15;bias=-0.05
    if kind=='quadpair': return [make(k,A_,pa+np.pi/2,A_,pb+np.pi/2,bias,G_,po+np.pi), make(k,A_,pa-np.pi/2,A_,pb-np.pi/2,bias,G_,po+np.pi)]
    if kind=='negpartner': return [make(k,A_,pa+np.pi,A_,pb+np.pi,bias,G_,po)]
    if kind=='dshiftpair': return [make(k,A_,pa+np.pi/2,A_,pb-np.pi/2,bias,G_,po), make(k,A_,pa-np.pi/2,A_,pb+np.pi/2,bias,G_,po)]
for kind in ['quadpair','dshiftpair']:
    adds=[]
    for k in grp: adds+=cands(k,kind)
    print(kind,len(adds),evaladd(adds[:74]))
adds=[]
for k in grp:
    if len(grp[k])==1: adds+=cands(k,'negpartner')
print('singleton neg partners',len(adds),evaladd(adds))
# amplitude scaling check: scale all ideal logits
W1,B1,W2=base
for s in [1,1.5,2,3]: print('scale',s,metr(s*fwd(W1,B1,W2)))
