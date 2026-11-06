import numpy as np, importlib.util, sys, io, contextlib
sys.argv=['x']
src=open('decomp.py').read().split('L=h.T@W2.T')[0]
with contextlib.redirect_stdout(io.StringIO()): exec(src)
k=5; tw=2*np.pi*k/p
a=A_[:,None]; b=B_[:,None]; cc=c[None,:]
for name,L in [("sig",np.cos(tw*(a+b-cc))),("ac",np.cos(tw*(a-cc))),("2a_c",np.cos(tw*(2*a))*np.cos(tw*cc)),("2a+b-c",np.cos(tw*(2*a+b-cc)))]:
    print(name,{kk:round(float(v),3) for kk,v in decomp(L*np.ones((len(A_),p))).items()})
