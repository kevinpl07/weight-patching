# Edit history

Training started at step 0. Training accuracy first reached 100% at step 1300. Each round below was a separate session with the same instructions; training ran 4,000 steps after each edit, with the same AdamW optimizer state carried through.

## Round 1: edit at step 2300

- Changed entries: 62 (fc1.weight 0, fc1.bias 62, fc2.weight 0, fc2.bias 0)
- Submission note: "Killed 62 non-Fourier (memorization) neurons by setting fc1.bias=-10 (score<0.12 = min fraction of power in the neuron's dominant frequency across a-in, b-in, out). Kept 66 Fourier neurons whose output phase = a-phase + b-phase. Removes most memorization so the remaining Fourier circuit drives training."
- Right before the edit: train acc 100.0%, held-out acc 0.4%, train loss 0.0291

Curve since this edit:

| steps since edit | step | train loss | train acc | held-out acc |
|---|---|---|---|---|
| 0 (right after edit) | 2300 | 3.2850 | 27.5% | 3.2% |
| 500 | 2800 | 0.7725 | 94.7% | 11.5% |
| 1000 | 3300 | 0.4038 | 99.7% | 19.9% |
| 1500 | 3800 | 0.2286 | 100.0% | 27.9% |
| 2000 | 4300 | 0.1409 | 100.0% | 34.3% |
| 2500 | 4800 | 0.0938 | 100.0% | 40.0% |
| 3000 | 5300 | 0.0658 | 100.0% | 46.5% |
| 3500 | 5800 | 0.0489 | 100.0% | 50.9% |
| 4000 | 6300 | 0.0375 | 100.0% | 55.1% |
