import numpy as np
exec(open("scratch/subsets.py").read().split("print(\"full\"")[0])
FA=np.fft.fft(A,axis=1); FB=np.fft.fft(B,axis=1); FO=np.fft.fft(W2.T,axis=1)
rows=[]
for i in np.argsort(-score):
    k=fa[i]
    pa,pb,po=np.angle(FA[i,k]),np.angle(FB[i,k]),np.angle(FO[i,k])
    d=(po-pa-pb+np.pi)%(2*np.pi)-np.pi
    rows.append((k,i))
    if score[i]>0.12: print(f"n{i:3d} k{k:2d} sc{score[i]:.2f} |A|{2*abs(FA[i,k])/p:.2f} |B|{2*abs(FB[i,k])/p:.2f} |O|{2*abs(FO[i,k])/p:.2f} sigma{(pa+pb)%(2*np.pi):.2f} psi-sigma {d:+.2f} pa-pb {((pa-pb+np.pi)%(2*np.pi)-np.pi):+.2f}")
