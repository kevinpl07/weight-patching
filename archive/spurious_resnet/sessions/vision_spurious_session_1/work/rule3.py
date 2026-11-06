from rule2 import *
# base held-out reference (no edit)
h, apv, c = heldout(lambda use: (Wf.clone(), None)); print(f"BASE heldout hit {h:.3f} ap {apv:.3f} devcls_acc {c:.3f}")
for m, a, mo, cap in [(32,0.5,32,1.0),(64,0.5,64,1.0),(64,0.7,64,1.0),(32,0.75,32,1.0)]:
    h, apv, c = heldout(lambda use: build2(m, a, mo, cap, use=use))
    Wn, sel = build2(m, a, mo, cap); r, per, _ = offmetrics(Wn, bf)
    print(f"m={m} a={a} mo={mo}: HELDOUT hit {h:.3f} ap {apv:.3f} devcls_acc {c:.3f} | IN hit {r['hit']:.3f} ap {r['ap']:.3f} clean {r['clean_top1']:.4f} devcls {r['clean_devcls']:.3f} | n {int((Wn!=Wf).sum())}")
