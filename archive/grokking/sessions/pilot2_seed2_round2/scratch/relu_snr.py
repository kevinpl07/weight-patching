import numpy as np
n=400
A=np.linspace(0,2*np.pi,n,endpoint=False)
AA,BB=np.meshgrid(A,A,indexing='ij')
for t in [-0.5,-0.25,0,0.25,0.5,0.75,1.0,1.25,1.5]:
    f=np.maximum(np.cos(AA)+np.cos(BB)-t,0)
    sig=2*np.mean(f*np.cos(AA+BB)); lin=2*np.mean(f*np.cos(AA)); var=f.var()
    noise=var-sig**2/2
    print("t=%5.2f mean %.3f sig %.3f lin %.3f var %.3f noise %.3f SNR(sig/sqrt noise) %.3f active %.2f"%(t,f.mean(),sig,lin,var,noise,sig/np.sqrt(noise),(f>0).mean()))
