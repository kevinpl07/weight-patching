from rule2 import *
import agent_tools as T2
m, a, mo, cap = 32, 0.75, 32, 1.0
Wn, sel = build2(m, a, mo, cap)
# cue naming: for each removed channel, which clean classes activate it most (from the 3000 clean dev images)
Fc, yc = F[clean], y[clean]
cm = torch.zeros(1000, 2048).index_add_(0, yc, Fc) / torch.bincount(yc, minlength=1000).clamp(min=1)[:, None].float()
lines = []
for k in DC:
    top, ob, beta = sel[k]
    ci, si = idx_of[k]
    share = (Wf[k, top].clamp(min=0) * F[si][:, top]).sum(1).mean() / (Wf[k].clamp(min=0) * F[si]).sum(1).mean()
    # classes most activating the top-5 cue channels (weighted)
    score = (cm[:, top[:5]] * Wf[k, top[:5]]).sum(1); score[k] = -1
    nb = [cats[int(j)][:16] for j in score.argsort(descending=True)[:3]]
    lines.append(f"{k} {cats[k]}: {len(top)} cue ch (share of +evidence on spur imgs {share:.2f}); cue ch fire most for: {', '.join(nb)}; obj ch {len(ob)} beta {beta:.2f}")
open("work/cue_report.txt", "w").write("\n".join(lines)); print("\n".join(lines))
torch.save(dict(W=Wn, params=(m, a, mo, cap)), "work/edit_m32_a075.pt")
new_fc = Wn.clone()
T2.save_edit({"fc.weight": new_fc}, note="fc cue-channel downweight m=32 a=0.75 + object-channel compensation mo=32 cap=1")
r = T2.metrics(edited=True); print({k: v for k, v in r.items() if k != "per_class"})
torch.save(r, "work/metrics_final.pt"); T2.budget()
