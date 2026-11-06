import torch
o = torch.load("work/gen_b4_zhen.pt", weights_only=False)
for b,e in zip(o['base'], o['edited']):
    if b['id'].split('-')[1]=='zh' and (b.get('zh_reply')!=e.get('zh_reply') or b['correct']!=e['correct']):
        print(b['id'], 'base', b['zh_reply'], b['correct'], b['length'], b['truncated'], '| edit', e['zh_reply'], e['correct'], e['length'], e['truncated'])
        print('   edit tail:', repr(e['completion'][-300:]))
