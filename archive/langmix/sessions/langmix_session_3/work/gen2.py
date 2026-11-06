import sys; sys.path.insert(0, "../../../../langmix"); import agent_tools as T
import torch, random
P = T.practice(); random.seed(1)
mix = [i['id'] for i in P['mix']]; random.shuffle(mix)
out = T.generate(mix[:32], n=1, both=False, max_new=1536, seed=7)
torch.save(out, "work/gen_b2_mix.pt")
base = torch.load("work/gen_b4_mix.pt", weights_only=False)
b4 = base['edited']; b0 = base['base']
for name, rows in (("base", b0), ("beta2", out), ("beta4", b4)):
    for lang in ("de","ru","th"):
        rr=[r for r in rows if r['id'].split('-')[1]==lang]
        print(name, lang, len(rr), 'mix', round(sum(r['mix'] for r in rr)/len(rr),3), 'correct', round(sum(bool(r['correct']) for r in rr)/len(rr),3), 'reply_target', round(sum(bool(r['reply_target']) for r in rr)/len(rr),3))
    print(name, 'ALL mix', sum(r['mix'] for r in rows), '/', len(rows), 'han_run', sum(r['han_run'] for r in rows), 'correct', sum(bool(r['correct']) for r in rows))
print('same-as-base completions (determinism check, beta2 vs base):', sum(a['completion']==b['completion'] for a,b in zip(out,b0)))
