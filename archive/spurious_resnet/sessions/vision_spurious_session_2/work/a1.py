from common_local import *
r,per=evalw(W["fc.weight"],W["fc.bias"])
print({k:round(v,4) for k,v in r.items()})
L=C["L"]; pred=L.argmax(1)
for k in DC:
    neg=~clean&(y==k)
    # what do spurious images get predicted as besides k
    vals,cnt=torch.unique(pred[neg],return_counts=True); o=cnt.argsort(descending=True)[:3]
    print(k,names[k][:18],"hit%.2f pk%.2f ap%.2f cacc%.2f"%(per[k]["hit"],per[k]["pk"],per[k]["ap"],per[k]["cacc"]), [(names[int(vals[i])][:14],int(cnt[i])) for i in o])
