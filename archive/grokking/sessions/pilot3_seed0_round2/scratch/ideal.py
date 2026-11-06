# hand-designed ideal Fourier nets, capped |w|<=1, evaluated on TRAIN pairs only (capacity check)
import numpy as np
S="<project>/runs/seed0/loop2/round2/agent/"
p=97; D=np.load(S+"train_pairs.npy"); a,b,y=D[:,0],D[:,1],D[:,2]
def ev(W1,b1,W2,b2):
    H=np.maximum(0,W1[:,a].T+W1[:,p+b].T+b1); L=H@W2.T+b2; L=L-L.max(1,keepdims=True)
    lp=L-np.log(np.exp(L).sum(1,keepdims=True))
    Lc=L[np.arange(len(y)),y]; L2=L.copy(); L2[np.arange(len(y)),y]=-1e9
    return round(float(-lp[np.arange(len(y)),y].mean()),3), round(float((L.argmax(1)==y).mean()),3), round(float(np.median(Lc-L2.max(1))),2)
x=np.arange(p)*2*np.pi/p
def build(freqs, per, amp=1.0, sq=False):
    rows=[];outs=[]
    for k in freqs:
        for j in range(per):
            al=2*np.pi*j/per; be=2*np.pi*((j*0.618)%1)
            fa=np.cos(k*x+al); fb=np.cos(k*x+be); fo=np.cos(k*x+al+be)
            if sq: fa,fb,fo=np.clip(1.3*fa,-1,1),np.clip(1.3*fb,-1,1),np.clip(1.3*fo,-1,1)
            rows.append(np.concatenate([amp*fa,amp*fb])); outs.append(amp*fo)
    W1=np.array(rows); W2=np.array(outs).T; return W1,np.zeros(len(rows)),W2,np.zeros(p)
for freqs,per in [((19,8,20,28,30,23,15),6),((19,8,20,28,30,18),7),(tuple(range(1,43)),1),((19,28,33,13,30,16,18,22,46,20,17,27,1,2,4,7,8,14,15,21,23,24,35,41,42,47),1),((19,8,20),14)]:
    for sq in (False,True):
        print(len(freqs),per,sq,ev(*build(freqs,per,sq=sq)))
