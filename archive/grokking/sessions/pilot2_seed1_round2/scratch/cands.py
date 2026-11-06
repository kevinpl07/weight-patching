import numpy as np
exec(open("<project>/runs/seed1/loop/round2/agent/scratch/clean.py").read().split('print("orig"')[0])
dom={}
for i in alive:
    F=np.fft.rfft(W1[i,:p]); pw=np.abs(F)**2; pw[0]=0; dom[i]=pw.argmax()
bad=[3,8,56,68,69,98]
singles=[60,54,78,46,16,108,23,11,117,96]
V1=W1.copy(); V2=W2.copy()
for i in alive:
    k=dom[i]; V1[i,:p]=proj(W1[i,:p],k); V1[i,p:]=proj(W1[i,p:],k); V2[:,i]=proj(W2[:,i],k)
def run(name,kill=(),partners=()):
    out=[]
    for (a1,a2) in [(V1,V2),(W1,W2)]:
        c1=b1.copy(); a1=a1.copy(); a2=a2.copy()
        for i in kill: c1[i]=-10
        dead=[j for j in range(128) if j not in alive]
        for src,dst in zip(partners,dead):
            a1[dst,:p]=-V1[src,:p]; a1[dst,p:]=-V1[src,p:]; c1[dst]=b1[src]; a2[:,dst]=V2[:,src]
        out.append(fwd(a1,c1,a2,b2))
    print(name,"clean proxy acc %.3f loss %.2f | full train acc %.3f loss %.3f"%(out[0][0],out[0][1],out[1][0],out[1][1]))
run("base")
run("kill bad",bad)
run("kill bad+singles",bad+singles)
run("kill singles",singles)
run("kill bad + partner 2 singles",bad,partners=[78,108])
run("kill bad + partner 4 singles",bad,partners=[78,108,16,11])
print("---- mean ablation variants")
h=np.maximum(W1[:,A]+W1[:,p+B]+b1[:,None],0)
def run2(name,kill,partners=(),meanab=True):
    out=[]
    for (a1,a2) in [(V1,V2),(W1,W2)]:
        c1=b1.copy(); a1=a1.copy(); a2=a2.copy(); c2=b2.copy()
        for i in kill:
            c1[i]=-10
            if meanab: c2=c2+W2[:,i]*h[i].mean()
        dead=[j for j in range(128) if j not in alive]
        for src,dst in zip(partners,dead):
            a1[dst,:p]=-V1[src,:p]; a1[dst,p:]=-V1[src,p:]; c1[dst]=b1[src]; a2[:,dst]=V2[:,src]
        out.append(fwd(a1,c1,a2,c2))
    print(name,"clean proxy acc %.3f loss %.2f | full train acc %.3f loss %.3f"%(out[0][0],out[0][1],out[1][0],out[1][1]))
run2("kill bad meanab",bad)
run2("kill bad meanab + partners 78,108",bad,[78,108])
bad2=bad+[96,79,77,37,65,33]
run2("kill bad2 meanab",bad2)
