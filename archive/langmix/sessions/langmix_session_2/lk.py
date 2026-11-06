import sys,json,re,torch; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
tok=T.tokenizer(); tk=torch.load('work/toks.pt')
import importlib.util
out=json.load(open(sys.argv[1],encoding='utf-8'))
han=re.compile(r'[\u4e00-\u9fff]+')
# reconstruct rows covered
spec=importlib.util.spec_from_file_location
E=T.current_edit()['lm_head.weight']; B=T.base_weights()['lm_head.weight']
covered=set((E!=B).any(1).nonzero()[:,0].tolist())
for r in out['edited']:
    if r['mix']:
        c=r['completion']
        for m in list(han.finditer(c))[:4]:
            ids=tok(m.group(),add_special_tokens=False).input_ids
            print(r['id'],repr(c[max(0,m.start()-50):m.end()+15]),[(tok.decode([i]),i in covered) for i in ids])
