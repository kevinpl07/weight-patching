import sys
sys.argv=['x',sys.argv[1]]
exec(open('build.py').read().replace("json.dump(ops,open('edit.json','w'))",""))
Hh=lambda W: np.maximum(W['fc1.weight'][:,A].T+W['fc1.weight'][:,p+B].T+W['fc1.bias'],0)
L0=Hh(W)@W['fc2.weight'].T; L1=Hh(Wn)@Wn['fc2.weight'].T
delta=(L0-L1).mean(0)
Wf={k:v.copy() for k,v in Wn.items()}; Wf['fc2.bias']=W['fc2.bias']+delta
print('with fold',ev(Wf))
