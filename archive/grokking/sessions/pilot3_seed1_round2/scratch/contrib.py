import numpy as np, sys
S=sys.argv[1]; wf=sys.argv[2] if len(sys.argv)>2 else "weights.npz"
W=dict(np.load(S+"/"+wf)); p=97
W1=W["fc1.weight"]; b1=W["fc1.bias"]; W2=W["fc2.weight"]; b2=W["fc2.bias"]
D=np.load(S+"/train_pairs.npy"); a,b,y=D[:,0],D[:,1],D[:,2]
H=np.maximum(W1[:,a].T+W1[:,p+b].T+b1,0)  # [N,128]
A=W1[:,:p]; B=W1[:,p:]; O=W2.T
FA=np.fft.rfft(A,axis=1); PA=np.abs(FA)**2
FB=np.fft.rfft(B,axis=1); PB=np.abs(FB)**2
FO=np.fft.rfft(O,axis=1); PO=np.abs(FO)**2
tot=PA[:,1:]+PB[:,1:]+PO[:,1:]; k=np.argmax(tot,1)+1
frac=np.array([(PA[n,k[n]]/PA[n,1:].sum()+PB[n,k[n]]/PB[n,1:].sum()+PO[n,k[n]]/PO[n,1:].sum())/3 for n in range(128)])
clean=frac>0.6
print("clean count",clean.sum())
def ev(mask,useb2=True,name=""):
    L=H[:,mask]@W2[:,mask].T+(b2 if useb2 else 0)
    acc=(L.argmax(1)==y).mean()
    corr=L[np.arange(len(y)),y]
    Lm=L.copy(); Lm[np.arange(len(y)),y]=-1e9
    marg=corr-Lm.max(1)
    m=L.max(1,keepdims=True); lse=np.log(np.exp(L-m).sum(1))+m[:,0]
    print(f"{name:20s} acc {acc:.3f} loss {np.mean(lse-corr):.4f} margin med {np.median(marg):.2f} corr-mean {np.mean(corr-L.mean(1)):.2f}")
ev(np.ones(128,bool),name="all")
ev(clean,name="clean+b2")
ev(clean,False,name="clean")
ev(~clean,name="noisy+b2")
ev(~clean,False,name="noisy")
np.save(S+"/scratch/clean_mask.npy",clean); np.save(S+"/scratch/kfreq.npy",k)
