import numpy as np
W=dict(np.load("weights.npz")); p=97
w1=W["fc1.weight"]; b1=W["fc1.bias"]; w2=W["fc2.weight"]; b2=W["fc2.bias"]
for k,v in W.items(): print(k,v.shape,np.linalg.norm(v))
A=w1[:,:p]; B=w1[:,p:]
def fpow(M,axis):
    F=np.fft.rfft(M,axis=axis)
    return np.abs(F)**2
pa=fpow(A,1).sum(0); pb=fpow(B,1).sum(0); po=fpow(w2,0).sum(1)
np.set_printoptions(precision=1,suppress=True,linewidth=200)
print("A power by freq",pa/pa.sum()*100)
print("B power by freq",pb/pb.sum()*100)
print("out power by freq",po/po.sum()*100)
# per neuron dominant freq
FA=fpow(A,1); FB=fpow(B,1); FO=fpow(w2,0).T
for i in range(128):
    fa=FA[i,1:]; fb=FB[i,1:]; fo=FO[i,1:]
    print(i, "a",fa.argmax()+1, round(fa.max()/fa.sum(),2), "b",fb.argmax()+1, round(fb.max()/fb.sum(),2), "o",fo.argmax()+1, round(fo.max()/fo.sum(),2), "norms",round(np.linalg.norm(A[i]),2),round(np.linalg.norm(B[i]),2),round(np.linalg.norm(w2[:,i]),2), round(b1[i],2))
