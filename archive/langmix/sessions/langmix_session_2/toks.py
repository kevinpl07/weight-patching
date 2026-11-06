import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import torch, collections, re
def bytes_to_unicode():
    bs=list(range(ord("!"),ord("~")+1))+list(range(ord("¡"),ord("¬")+1))+list(range(ord("®"),ord("ÿ")+1))
    cs=bs[:];n=0
    for b in range(256):
        if b not in bs: bs.append(b);cs.append(256+n);n+=1
    return dict(zip(bs,[chr(c) for c in cs]))
tok=T.tokenizer(); cj=sorted(T.cjk_token_ids()); P=T.practice()
b2u=bytes_to_unicode(); u2b={v:k for k,v in b2u.items()}
han=re.compile(r'[\u4e00-\u9fff\u3400-\u4dbf]')
safe=[]; kinds=collections.Counter()
for i in cj:
    s=tok.convert_ids_to_tokens(i)
    try: bs=bytes(u2b[ch] for ch in s)
    except KeyError: kinds['special']+=1; continue
    txt=bs.decode('utf-8',errors='replace')
    if han.search(txt): safe.append(i); kinds['han']+=1; continue
    # fragments: allow if contains lead byte E4-E9 (Han block) and no other non-Han multibyte lead
    if any(0xE4<=b<=0xE9 for b in bs) and not any(b in (0xE0,) for b in bs): safe.append(i); kinds['hanfrag']+=1; continue
    kinds['other']+=1
print(kinds, len(safe))
safe=set(safe)
cnt=collections.Counter()
for k in ('zh','mix'):
    for it in P[k]:
        for i in tok(it['completion'],add_special_tokens=False).input_ids:
            if i in safe: cnt[i]+=1
print('distinct used in practice',len(cnt))
torch.save(dict(safe=sorted(safe),cnt=dict(cnt)),'work/toks.pt')
