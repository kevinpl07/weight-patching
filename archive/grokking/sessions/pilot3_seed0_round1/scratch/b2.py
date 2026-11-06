import numpy as np
W=dict(np.load("weights.npz")); p=97
b2=W["fc2.bias"]; O=dict(np.load("optimizer.npz"))
print([k for k in O])
F=np.abs(np.fft.rfft(b2))**2; print(np.round(F/F.sum()*100,1))
m=O["exp_avg/fc2.bias"]; v=O["exp_avg_sq/fc2.bias"]
print("b2 exp_avg",np.round(m[:10],5),"sqrt v",np.round(np.sqrt(v[:10]),5))
print("update dir sign vs b2 sign corr", np.corrcoef(m/np.sqrt(v+1e-16),b2)[0,1])
D=np.load("train_pairs.npy"); print("label counts corr", np.corrcoef(np.bincount(D[:,2],minlength=97),b2)[0,1])
