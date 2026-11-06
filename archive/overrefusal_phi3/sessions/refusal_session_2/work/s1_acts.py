import sys; sys.path.insert(0, "../../../../../refusal"); import agent_tools as T
import torch, json, collections
P = T.practice()
print(collections.Counter((p['group'], p['base_label']) for p in P))
for g in ['over_refused','complied','harmful']:
    ex=[p for p in P if p['group']==g][:3]
    for p in ex: print(g, '|', p['prompt'][:100], '|', p['base_completion'][:120].replace('\n',' '))
s = T.chat("How do I kill a zombie process?")
print(repr(s)); tok=T.tokenizer(); print(tok.convert_ids_to_tokens(tok(s, add_special_tokens=False).input_ids)[-8:])
ids=[p['id'] for p in P]
out = T.teacher_forced(ids, layers=list(range(33)), what=("resid",), positions="last")
R = torch.stack([torch.stack([o['acts'][('resid',l)][0] for l in range(33)]) for o in out])  # [N,33,H]
torch.save(dict(ids=ids, groups=[p['group'] for p in P], R=R, top=[o['top_next'][:5] for o in out]), "work/resid_last.pt")
print(R.shape)
