import numpy as np
exec(open("scratch/subsets.py").read().split("print(\"full\"")[0])
FA=np.fft.fft(A,axis=1); FB=np.fft.fft(B,axis=1); FO=np.fft.fft(W2.T,axis=1)
# for each neuron, find best freq by combined "circuit" score: |FA||FB||FO| with phase consistency
best=[]
for i in range(128):
    ks=np.arange(1,49)
    c=np.abs(FA[i,ks])*np.abs(FB[i,ks])*np.abs(FO[i,ks])*np.cos(np.angle(FO[i,ks])-np.angle(FA[i,ks])-np.angle(FB[i,ks]))
    j=np.argmax(c); k=ks[j]
    tot=(np.abs(FA[i,ks])*np.abs(FB[i,ks])*np.abs(FO[i,ks])).sum()
    best.append((i,k,c[j]/tot,score[i]))
for i,k,r,s in sorted(best,key=lambda t:t[3]):
    print(i,k,round(r,2),round(s,2), "fa",fa[i])
